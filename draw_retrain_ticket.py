"""
Script for running finetuning on glue tasks.

Largely copied from:
    https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
"""
import argparse
import logging
import os
import csv
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.utils.prune as prune
from transformers import (
    AdamW, AutoConfig, AutoTokenizer
)

import sys
sys.path.append("..")
sys.path.append("/root/RobustRepository")

from utils import Collator, Huggingface_dataset, ExponentialMovingAverage
from robust_ticket.model.masked_bert import MaskedBertForSequenceClassification

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument("--dataset_name", default='glue', type=str)
    parser.add_argument("--task_name", default=None, type=str)
    parser.add_argument('--ckpt_dir', type=Path, default=Path('/root/robust_transfer/saved_models/pruning/'))
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--valid', type=str, default='validation')  # test for imdb, agnews; validation for GLUEs
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=False)
    parser.add_argument('--do_lower_case', type=bool, default=True)

    # subnetwork pruning
    parser.add_argument('--sparsity', type=float, default=0.4)
    parser.add_argument('--masked_model_path', type=str, default='/root/save_models/robust_ticket/Search-Robust-Ticket_bert_glue-sst2_lr0.1_lambda0.5_adv-lr0.03_adv-step2_epochs20/epoch5')
    # (1,1) for weight masking  (768,1) for neuron masking  (768, 768) for layer masking
    parser.add_argument('--out_w_per_mask', type=int, default=1)
    parser.add_argument('--in_w_per_mask', type=float, default=1)
    parser.add_argument('--mask_p', type=float, default=0.9)  # init mask score

    # adversarial attack
    parser.add_argument('--do_attack', type=int, default=1)
    parser.add_argument("--num_examples", default=872, type=int)
    parser.add_argument('--result_file', type=str, default='attack_result.csv')

    # hyper-parameters
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--eval_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', default=1e-2, type=float)  # BERT default
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")  # BERT default
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Linear warmup over warmup_steps.")  # BERT default
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bias_correction', default=True)
    parser.add_argument('-f', '--force_overwrite', default=True)
    parser.add_argument('--debug', action='store_true')

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


def adversarial_attack(output_dir, args):

    for epoch in range(args.epochs):

        attack_path = Path(str(output_dir) + '/epoch' + str(epoch))
        original_accuracy, accuracy_under_attack, attack_succ = attack_test(attack_path, args)

        out_csv = open(args.result_file, 'a', encoding='utf-8', newline="")
        csv_writer = csv.writer(out_csv)
        csv_writer.writerow([attack_path, original_accuracy, accuracy_under_attack, attack_succ])
        out_csv.close()
    pass


def attack_test(attack_path, args):

    from textattack.attack_recipes.textfooler_jin_2019 import TextFoolerJin2019
    from textattack.datasets import HuggingFaceDataset
    from textattack.attack_results import SuccessfulAttackResult, MaximizedAttackResult, FailedAttackResult
    from textattack.models.wrappers.huggingface_model_wrapper import HuggingFaceModelWrapper
    from textattack import Attacker
    from textattack import AttackArgs

    # for model
    config = AutoConfig.from_pretrained(attack_path)
    model = MaskedBertForSequenceClassification.from_pretrained(
        attack_path, config=config, out_w_per_mask=args.out_w_per_mask,
        in_w_per_mask=args.in_w_per_mask, mask_p=args.mask_p)
    tokenizer = AutoTokenizer.from_pretrained(attack_path)
    model.eval()

    # for dataset
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    attack = TextFoolerJin2019.build(model_wrapper)
    dataset = HuggingFaceDataset(args.dataset_name, args.task_name, split=args.valid)

    # for attack
    attack_args = AttackArgs(num_examples=args.num_examples,
                             disable_stdout=True, random_seed=args.seed)
    attacker = Attacker(attack, dataset, attack_args)
    num_results = 0
    num_successes = 0
    num_failures = 0
    for result in attacker.attack_dataset():
        num_results += 1
        if (
                type(result) == SuccessfulAttackResult
                or type(result) == MaximizedAttackResult
        ):
            num_successes += 1
        if type(result) == FailedAttackResult:
            num_failures += 1

    original_accuracy = (num_successes + num_failures) * 100.0 / num_results
    accuracy_under_attack = num_failures * 100.0 / num_results

    if original_accuracy != 0:
        attack_succ = (original_accuracy - accuracy_under_attack) * 100.0 / original_accuracy
    else:
        attack_succ = 0

    return original_accuracy, accuracy_under_attack, attack_succ


def positive_mask_scores(model):
    # transform mask_scores to a positive value and then it used for pruning.
    for ii in range(12):
        # query
        module = model.bert.encoder.layer[ii].attention.self.query.mask.mask_scores
        module.data = torch.sigmoid(module.data)
        # key
        module = model.bert.encoder.layer[ii].attention.self.key.mask.mask_scores
        module.data = torch.sigmoid(module.data)
        # value
        module = model.bert.encoder.layer[ii].attention.self.value.mask.mask_scores
        module.data = torch.sigmoid(module.data)
        # attention output dense
        module = model.bert.encoder.layer[ii].attention.output.dense.mask.mask_scores
        module.data = torch.sigmoid(module.data)
        # intermediate dense
        module = model.bert.encoder.layer[ii].intermediate.dense.mask.mask_scores
        module.data = torch.sigmoid(module.data)
        # output dense
        module = model.bert.encoder.layer[ii].output.dense.mask.mask_scores
        module.data = torch.sigmoid(module.data)
    # output dense
    module = model.bert.pooler.dense.mask.mask_scores
    module.data = torch.sigmoid(module.data)


def pruning_mask_score(model, px):
    """
    Pruning mask score;
    mask score will be translated to mask through

    :param model:
    :param px: sparsity of mask
    :return:
    """

    parameters_to_prune = []
    for ii in range(12):
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.query.mask, 'mask_scores'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.key.mask, 'mask_scores'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.self.value.mask, 'mask_scores'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].attention.output.dense.mask, 'mask_scores'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].intermediate.dense.mask, 'mask_scores'))
        parameters_to_prune.append((model.bert.encoder.layer[ii].output.dense.mask, 'mask_scores'))

    parameters_to_prune.append((model.bert.pooler.dense.mask, 'mask_scores'))
    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )


def draw_ticket_mask(model, sparsity):
    # draw masks of the robust ticket with a certain sparsity
    positive_mask_scores(model)
    pruning_mask_score(model, px=sparsity)
    mask_scores_mask_dict = {}
    model_dict = model.state_dict()
    for key in model_dict.keys():
        if 'mask_scores_mask' in key:
            mask_scores_mask_dict[key] = model_dict[key]

    return mask_scores_mask_dict


def init_mask_score(model, ticket_mask):
    for ii in range(12):
        # query
        module = model.bert.encoder.layer[ii].attention.self.query.mask.mask_scores
        module_mask = 'bert.encoder.layer.{}.attention.self.query.mask.mask_scores_mask'.format(ii)
        mask = ticket_mask[module_mask]
        module.data = 20*mask-20*(1-mask)
        # key
        module = model.bert.encoder.layer[ii].attention.self.key.mask.mask_scores
        module_mask = 'bert.encoder.layer.{}.attention.self.key.mask.mask_scores_mask'.format(ii)
        mask = ticket_mask[module_mask]
        module.data = 20*mask-20*(1-mask)
        # value
        module = model.bert.encoder.layer[ii].attention.self.value.mask.mask_scores
        module_mask = 'bert.encoder.layer.{}.attention.self.value.mask.mask_scores_mask'.format(ii)
        mask = ticket_mask[module_mask]
        module.data = 20*mask-20*(1-mask)
        # attention output dense
        module = model.bert.encoder.layer[ii].attention.output.dense.mask.mask_scores
        module_mask = 'bert.encoder.layer.{}.attention.output.dense.mask.mask_scores_mask'.format(ii)
        mask = ticket_mask[module_mask]
        module.data = 20*mask-20*(1-mask)
        # intermediate dense
        module = model.bert.encoder.layer[ii].intermediate.dense.mask.mask_scores
        module_mask = 'bert.encoder.layer.{}.intermediate.dense.mask.mask_scores_mask'.format(ii)
        mask = ticket_mask[module_mask]
        module.data = 20*mask-20*(1-mask)
        # output dense
        module = model.bert.encoder.layer[ii].output.dense.mask.mask_scores
        module_mask = 'bert.encoder.layer.{}.output.dense.mask.mask_scores_mask'.format(ii)
        mask = ticket_mask[module_mask]
        module.data = 20*mask-20*(1-mask)
    # output dense
    module = model.bert.pooler.dense.mask.mask_scores
    module_mask = 'bert.pooler.dense.mask.mask_scores_mask'
    mask = ticket_mask[module_mask]
    module.data = 20 * mask - 20 * (1 - mask)
    pass


def main(args):
    set_seed(args.seed)

    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news':
        output_dir = Path(os.path.join(args.ckpt_dir, 'Draw-Retrain_{}_{}_lr{}_sparsity{}_epochs{}'
                                       .format(args.model_name, args.dataset_name,
                                               args.lr, args.sparsity, args.epochs)))
    else:
        output_dir = Path(os.path.join(args.ckpt_dir, 'Draw-Retrain_{}_{}-{}_lr{}_sparsity{}_epochs{}'
                                       .format(args.model_name, args.dataset_name,
                                               args.task_name, args.lr, args.sparsity, args.epochs)))

    if not output_dir.exists():
        logger.info(f'Making checkpoint directory: {output_dir}')
        output_dir.mkdir(parents=True)
    elif not args.force_overwrite:
        raise RuntimeError('Checkpoint directory already exists.')
    log_file = os.path.join(output_dir, 'INFO.log')
    logger.addHandler(logging.FileHandler(log_file))

    # Load masked pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=args.do_lower_case)
    model = MaskedBertForSequenceClassification.from_pretrained(
        args.masked_model_path, config=config, out_w_per_mask=args.out_w_per_mask,
        in_w_per_mask=args.in_w_per_mask, mask_p=args.mask_p)
    model.to(device)

    # draw robust tickets
    ticket_mask = draw_ticket_mask(model, args.sparsity)

    # reload pre-trained model
    model = MaskedBertForSequenceClassification.from_pretrained(
        args.model_name, config=config, out_w_per_mask=args.out_w_per_mask,
        in_w_per_mask=args.in_w_per_mask, mask_p=args.mask_p)
    model.to(device)
    init_mask_score(model, ticket_mask)

    # dataset
    collator = Collator(pad_token_id=tokenizer.pad_token_id)
    # for training
    train_dataset = Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name, subset=args.task_name)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    # for dev
    dev_dataset = Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                            subset=args.task_name, split=args.valid)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)
    # for test
    # if args.do_test:
    #     test_dataset = Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
    #                                              subset=args.task_name, split='test')
    #     test_loader = DataLoader(test_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_masked_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if 'mask_score' not in n and p.requires_grad and
                       not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.lr,
        },
        {"params": [p for n, p in model.named_parameters() if 'mask_score' not in n and p.requires_grad and
                    any(nd in n for nd in no_decay)],
         "weight_decay": 0.0,
         "lr": args.lr,
         },

    ]
    optimizer = AdamW(
        optimizer_masked_parameters,
        eps=args.adam_epsilon,
        correct_bias=args.bias_correction
    )

    # Use suggested learning rate scheduler
    num_training_steps = len(train_dataset) * args.epochs // args.bsz
    warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

    print('Before training....')
    logger.info(
                f'Pct_binary: {model.compute_binary_pct(): 0.4f}, '
                f'Pct_Less0.5: {model.compute_half_pct(): 0.4f} '
                )

    processed = 0
    iteration_step = 0
    best_accuracy = 0
    for epoch in range(args.epochs):
        logger.info('Training...')
        model.train()
        avg_loss = ExponentialMovingAverage()
        pbar = tqdm(train_loader)
        for model_inputs, labels in pbar:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            iteration_step += 1
            labels = labels.to(device)
            model.zero_grad()
            logits = model(**model_inputs).logits
            loss = F.cross_entropy(logits, labels.squeeze(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            avg_loss.update(loss.item())
            pbar.set_description(f'epoch: {epoch: d}, '
                                 f'loss: {avg_loss.get_metric(): 0.4f}, '
                                 f'lr: {optimizer.param_groups[0]["lr"]: .3e}')
        s = Path(str(output_dir) + '/epoch' + str(epoch))
        if not s.exists():
            s.mkdir(parents=True)
        model.save_pretrained(s)
        tokenizer.save_pretrained(s)
        torch.save(args, os.path.join(s, "training_args.bin"))

        # logger.info('Evaluating...')
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
                    f'Pct_binary: {model.compute_binary_pct(): 0.4f}, '
                    f'Pct_Less0.5: {model.compute_half_pct(): 0.4f} '
                    )

        if accuracy > best_accuracy:
            logger.info('Best performance so far.')
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            best_accuracy = accuracy
            best_dev_epoch = epoch
    logger.info(f'Best dev metric: {best_accuracy} in Epoch: {best_dev_epoch}')

    if args.do_attack:
        adversarial_attack(output_dir, args)


    # test using best model
    # if args.do_test:
    #     logger.info('Testing...')
    #     model = AutoModelForSequenceClassification.from_pretrained(output_dir, config=config)
    #
    #     model.eval()
    #     correct = 0
    #     total = 0
    #     with torch.no_grad():
    #         for model_inputs, labels in test_loader:
    #             model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    #             labels = labels.to(device)
    #             logits, *_ = model(**model_inputs)
    #             _, preds = logits.max(dim=-1)
    #             correct += (preds == labels.squeeze(-1)).sum().item()
    #             total += labels.size(0)
    #         accuracy = correct / (total + 1e-13)
    #     logger.info(f'Accuracy: {accuracy : 0.4f}')


if __name__ == '__main__':

    args = parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)
