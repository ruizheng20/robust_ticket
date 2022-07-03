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
import csv
from tqdm import tqdm
import sys


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
)

from utils import Collator, Huggingface_dataset, ExponentialMovingAverage

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args():
    parser = argparse.ArgumentParser()
    # settings
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--dataset_name', default='glue', type=str)
    parser.add_argument('--task_name', default=None, type=str)
    parser.add_argument('--ckpt_dir', type=Path, default=Path('./saved_models/fine-tune/'))
    parser.add_argument('--num_labels', type=int, default=2)
    parser.add_argument('--do_lower_case', type=bool, default=True)

    # adversarial attack
    parser.add_argument('--num_examples', default=200, type=int)
    parser.add_argument('--result_file', type=str, default='attack_result.csv')

    # hyper-parameters
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--eval_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', default=1e-2, type=float)  # BERT default
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help="Epsilon for Adam optimizer.")  # BERT default
    parser.add_argument('--warmup_ratio', default=0.1, type=float,
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
    model = AutoModelForSequenceClassification.from_pretrained(attack_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(attack_path)
    model.eval()

    # for dataset
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    attack = TextFoolerJin2019.build(model_wrapper)

    if args.dataset_name in ['imdb', 'ag_news']:
        attack_valid = 'test'
    elif args.task_name == 'mnli':
        attack_valid = 'validation_matched'
    else:
        attack_valid = 'validation'

    dataset = HuggingFaceDataset(args.dataset_name, args.task_name, split=attack_valid)

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


def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    avg_loss = ExponentialMovingAverage()
    with torch.no_grad():
        for model_inputs, labels in data_loader:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            logits = model(**model_inputs).logits
            _, preds = logits.max(dim=-1)
            loss = F.cross_entropy(logits, labels.squeeze(-1))
            avg_loss.update(loss.item())
            correct += (preds == labels.squeeze(-1)).sum().item()
            total += labels.size(0)
        accuracy = correct / (total + 1e-13)
    return accuracy, avg_loss.get_metric()


def main(args):
    set_seed(args.seed)

    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news':
        output_dir = Path(os.path.join(args.ckpt_dir, 'finetune_{}_{}_lr{}_epochs{}_seed{}'
                                       .format(args.model_name, args.dataset_name,
                                               args.lr, args.epochs, args.seed)))
    else:
        output_dir = Path(os.path.join(args.ckpt_dir, 'finetune_{}_{}-{}_lr{}_epochs{}_seed{}'
                                       .format(args.model_name, args.dataset_name,
                                               args.task_name, args.lr, args.epochs, args.seed)))

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
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
    model.to(device)

    train_dataset, train_loader, dev_loader, test_loader = load_data(tokenizer, args)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        eps=args.adam_epsilon,
        correct_bias=args.bias_correction
    )

    # Use suggested learning rate scheduler
    num_training_steps = len(train_dataset) * args.epochs // args.bsz
    warmup_steps = num_training_steps * args.warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

    best_dev_epoch, best_dev_accuracy, test_accuracy = 0, 0, 0

    for epoch in range(args.epochs):
        model.train()
        avg_loss = ExponentialMovingAverage()
        pbar = tqdm(train_loader)

        for model_inputs, labels in pbar:

            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
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
        torch.save(args, os.path.join(s, 'training_args.bin'))

        # test after one epoch
        dev_accuracy, dev_loss = evaluate(model, dev_loader, device)
        logger.info(f'Epoch: {epoch}, '
                    f'Loss: {avg_loss.get_metric(): 0.4f}, '
                    f'Lr: {optimizer.param_groups[0]["lr"]: .3e}, '
                    f'Dev_Accuracy: {dev_accuracy}')

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_dev_epoch = epoch
            test_accuracy, test_loss = evaluate(model, test_loader, device)
            logger.info(f'**** Test Accuracy: {test_accuracy}, Test_Loss: {test_loss}')
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))

    logger.info(f'**** Best dev metric: {best_dev_accuracy} in Epoch: {best_dev_epoch}')
    logger.info(f'**** Best Test metric: {test_accuracy} in Epoch: {best_dev_epoch}')

    last_test_accuracy, last_test_loss = evaluate(model, test_loader, device)
    logger.info(f'Last epoch test_accuracy: {last_test_accuracy}, test_loss: {last_test_loss}')

    # for adversarial robustness evaluation
    adversarial_attack(output_dir, args)


if __name__ == '__main__':

    args = parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)
