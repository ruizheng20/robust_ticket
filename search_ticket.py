"""
Script for running finetuning on glue tasks.

Largely copied from:
    https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
"""
import argparse
import logging
import os
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AdamW, AutoConfig, AutoTokenizer
)
from masked_bert import MaskedBertForSequenceClassification
from utils import Collator, Huggingface_dataset, ExponentialMovingAverage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--model_name', type=str, default='./your_fine-tune_path')
    parser.add_argument('--model_type', type=str, default='bert')
    parser.add_argument('--dataset_name', default='glue', type=str)
    parser.add_argument('--task_name', default=None, type=str)
    parser.add_argument('--ckpt_dir', type=Path, default=Path('./saved_models/search-ticket/'))
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--do_lower_case', type=bool, default=True)

    # mask learning
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lambda_amp', type=float, default=1)
    parser.add_argument('--lambda_init', type=float, default=0)
    parser.add_argument('--lambda_final', type=float, default=1)
    parser.add_argument('--lambda_startup_frac', type=float, default=0.1)
    parser.add_argument('--lambda_warmup_frac', type=float, default=0.3)
    # (1,1) for weight masking  (768,1) for neuron masking  (768, 768) for layer masking
    parser.add_argument('--out_w_per_mask', type=int, default=1)
    parser.add_argument('--in_w_per_mask', type=float, default=1)
    parser.add_argument('--mask_p', type=float, default=0.9)  # init mask score

    # hyper-parameters
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--eval_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)  # larger epoch
    parser.add_argument('--weight_decay', default=1e-6, type=float)  # Not BERT default
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Epsilon for Adam optimizer.')  # BERT default
    parser.add_argument('--warmup_ratio', default=0.1, type=float,
                        help='Linear warmup over warmup_steps.')  # BERT default
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bias_correction', default=True)
    parser.add_argument('-f', '--force_overwrite', default=True)
    parser.add_argument('--debug', action='store_true')

    # Adversarial training specific
    parser.add_argument('--adv_steps', default=2, type=int,
                        help='Number of gradient ascent steps for the adversary')
    parser.add_argument('--adv_lr', default=0.03, type=float,
                        help='Step size of gradient ascent')
    parser.add_argument('--adv_init_mag', default=0.05, type=float,
                        help='Magnitude of initial (adversarial?) perturbation')
    parser.add_argument('--adv_max_norm', default=0, type=float,
                        help='adv_max_norm = 0 means unlimited')
    parser.add_argument('--adv_norm_type', default='l2', type=str,
                        help='norm type of the adversary')
    parser.add_argument('--adv_change_rate', default=0.2, type=float,
                        help='change rate of a sentence')
    parser.add_argument('--max_grad_norm', default=1, type=float, help='max gradient norm')

    args = parser.parse_args()
    if args.ckpt_dir is not None:
        os.makedirs(args.ckpt_dir, exist_ok=True)
    else:
        args.ckpt_dir = '.'
    return args


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.

    From:
        https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def load_data(tokenizer, args):
    # dataloader
    collator = Collator(pad_token_id=tokenizer.pad_token_id)
    # for training and dev
    train_dataset = Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name, subset=args.task_name)

    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news':
        split_ratio = 0.1
        train_size = round(int(len(train_dataset) * (1 - split_ratio)))
        dev_size = int(len(train_dataset)) - train_size
        # train and dev dataloader
        train_dataset, dev_dataset = torch.utils.data.random_split(train_dataset, [train_size, dev_size])
        train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

        test_dataset = Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                 subset=args.task_name, split='test')
        test_loader = DataLoader(test_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)
    elif args.task_name == 'mnli':
        train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
        dev_dataset = Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                 subset=args.task_name, split='validation_matched')
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)
        test_loader = dev_loader
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
        dev_dataset = Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                 subset=args.task_name, split='validation')
        dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)
        test_loader = dev_loader

    return train_dataset, train_loader, dev_loader, test_loader


def main(args):
    logger.info(args)

    set_seed(args.seed)
    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news':
        output_dir = Path(
            os.path.join(args.ckpt_dir, 'search-robust-ticket_{}_{}_lr{}_lambda{}_adv-lr{}_adv-step{}_epochs{}'
                         .format(args.model_type, args.dataset_name,
                                 args.lr, args.lambda_amp,
                                 args.adv_lr, args.adv_steps, args.epochs)))
    else:
        output_dir = Path(os.path.join(args.ckpt_dir,
                                       'search-robust-ticket_{}_{}-{}_lr{}_lambda{}_adv-lr{}_adv-step{}_epochs{}'
                                       .format(args.model_type, args.dataset_name, args.task_name,
                                               args.lr, args.lambda_amp, args.adv_lr,
                                               args.adv_steps, args.epochs)))
    if not output_dir.exists():
        logger.info(f'Making checkpoint directory: {output_dir}')
        output_dir.mkdir(parents=True)
    elif not args.force_overwrite:
        raise RuntimeError('Checkpoint directory already exists.')
    log_file = os.path.join(output_dir, 'INFO.log')
    logger.addHandler(logging.FileHandler(log_file))

    # pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)
    model = MaskedBertForSequenceClassification.from_pretrained(
        args.model_name, config=config, out_w_per_mask=args.out_w_per_mask,
        in_w_per_mask=args.in_w_per_mask, mask_p=args.mask_p)  # Masked BERT
    model.to(device)

    train_dataset, train_loader, dev_loader, test_loader = load_data(tokenizer, args)

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer_masked_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if 'mask_score' in n and p.requires_grad],
            "weight_decay": args.weight_decay
        },
    ]
    optimizer = AdamW(
        optimizer_masked_parameters,
        lr=args.lr,
        eps=args.adam_epsilon,
        correct_bias=args.bias_correction
    )

    # Use suggested learning rate scheduler
    num_training_steps = len(train_dataset) * args.epochs // args.bsz
    warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

    best_accuracy, processed, iteration_step = 0, 0, 0
    for epoch in range(args.epochs):
        logger.info('Training...')
        model.train()
        avg_loss = ExponentialMovingAverage()

        pbar = tqdm(train_loader)
        for model_inputs, labels in pbar:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            optimizer.zero_grad()

            processed += len(model_inputs)
            iteration_step += 1

            # for freelb
            word_embedding_layer = model.get_input_embeddings()
            input_ids = model_inputs['input_ids']
            attention_mask = model_inputs['attention_mask']
            embedding_init = word_embedding_layer(input_ids)

            # initialize delta
            if args.adv_init_mag > 0:
                input_mask = attention_mask.to(embedding_init)
                input_lengths = torch.sum(input_mask, 1)
                if args.adv_norm_type == 'l2':
                    delta = torch.zeros_like(embedding_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                    dims = input_lengths * embedding_init.size(-1)
                    magnitude = args.adv_init_mag / torch.sqrt(dims)
                    delta = (delta * magnitude.view(-1, 1, 1))
                elif args.adv_norm_type == 'linf':
                    delta = torch.zeros_like(embedding_init).uniform_(-args.adv_init_mag,
                                                                      args.adv_init_mag) * input_mask.unsqueeze(2)
            else:
                delta = torch.zeros_like(embedding_init)

            total_loss = 0.0
            for astep in range(args.adv_steps):
                # (0) forward
                delta.requires_grad_()
                batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
                logits = model(**batch).logits

                # (1) backward
                losses = F.cross_entropy(logits, labels.squeeze(-1))
                loss = torch.mean(losses)
                loss = loss / args.adv_steps
                total_loss += loss.item()
                loss.backward()

                if astep == args.adv_steps - 1:
                    break

                # (2) get gradient on delta
                delta_grad = delta.grad.clone().detach()

                # (3) update and clip
                if args.adv_norm_type == "l2":
                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                    if args.adv_max_norm > 0:
                        delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                        exceed_mask = (delta_norm > args.adv_max_norm).to(embedding_init)
                        reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                        delta = (delta * reweights).detach()
                elif args.adv_norm_type == "linf":
                    denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1,
                                                                                                             1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                embedding_init = word_embedding_layer(input_ids)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # schedule lambda_reg - constant, then linear, then constant
            lambda_ratio = max(0, min(1, (processed - args.lambda_startup_frac * args.epochs * len(train_loader)) /
                                      (args.lambda_warmup_frac * args.epochs * len(train_loader))))
            lambda_reg = args.lambda_init + (args.lambda_final - args.lambda_init) * lambda_ratio
            lambda_reg *= args.lambda_amp
            reg = model.compute_total_regularizer()
            (lambda_reg * reg).backward()

            optimizer.step()
            scheduler.step()

            avg_loss.update(total_loss)
            pbar.set_description(f'Epoch: {epoch: d}, '
                                 f'Loss: {avg_loss.get_metric(): 0.4f}, '
                                 f'Reg: {reg.item(): 0.4f}, ' 
                                 f'Lr: {optimizer.param_groups[0]["lr"]: .3e}, ')

        s = Path(str(output_dir) + '/epoch' + str(epoch))
        if not s.exists():
            s.mkdir(parents=True)
        model.save_pretrained(s)
        tokenizer.save_pretrained(s)
        torch.save(args, os.path.join(s, 'training_args.bin'))

        logger.info('Evaluating...')
        model.eval()
        correct = 0
        total = 0
        total_loss = 0
        val_loss = ExponentialMovingAverage()
        with torch.no_grad():
            for model_inputs, labels in dev_loader:
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                labels = labels.to(device)
                logits = model(**model_inputs).logits
                _, preds = logits.max(dim=-1)
                losses = F.cross_entropy(logits, labels.squeeze(-1))
                loss = torch.mean(losses)
                total_loss += loss.item()
                val_loss.update(total_loss)
                correct += (preds == labels.squeeze(-1)).sum().item()
                total += labels.size(0)
            accuracy = correct / (total + 1e-13)
        logger.info(f'Epoch: {epoch}, '
                    f'Loss_train: {avg_loss.get_metric(): 0.4f}, '
                    f'Loss_val: {val_loss.get_metric(): 0.4f}, '
                    f'Lr: {optimizer.param_groups[0]["lr"]: .3e}, '
                    f'Accuracy: {accuracy}, '
                    f'Regularizer: {reg.item(): 0.4f}, '
                    f'Pct_binary: {model.compute_binary_pct(): 0.4f}, '
                    f'Pct_Less0.5: {model.compute_half_pct(): 0.4f}, '
                    )

        if accuracy > best_accuracy:
            logger.info('Best performance so far.')
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            best_accuracy = accuracy
            best_dev_epoch = epoch
    logger.info(f'Best dev metric: {best_accuracy} in Epoch: {best_dev_epoch}')


if __name__ == '__main__':

    args = parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)
