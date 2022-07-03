# coding=utf-8
"""
Attack Module
"""
import sys

import argparse
import csv
import logging
from transformers import AutoConfig, AutoTokenizer
from masked_bert import MaskedBertForSequenceClassification
from textattack import Attack
from textattack import Attacker
from textattack import AttackArgs
from textattack.attack_results import SuccessfulAttackResult, MaximizedAttackResult, FailedAttackResult
from textattack.attack_recipes import (PWWSRen2019,
                                       BAEGarg2019,
                                       TextBuggerLi2018,
                                       TextFoolerJin2019,
                                       )
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import HuggingFaceDataset
from textattack.constraints.pre_transformation import InputColumnModification
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.goal_functions import UntargetedClassification
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.transformations import WordSwapEmbedding

logger = logging.getLogger(__name__)


def build_default_attacker(args, model) -> Attack:
    attacker = None
    if args.attack_method == 'textbugger':
        attacker = TextBuggerLi2018.build(model)
    elif args.attack_method == 'textfooler':
        attacker = TextFoolerJin2019.build(model)
    elif args.attack_method == 'bertattack':
        attacker = BAEGarg2019.build(model)
    elif args.attack_method == 'pwws':
        attacker = PWWSRen2019.build(model)
    else:
        print("Not implement attck!")
        exit(41)
    input_column_modification0= InputColumnModification(["sentence1", "sentence2"], {"sentence1"})
    input_column_modification1 = InputColumnModification(["sentence", "question"], {"sentence"})
    attacker.pre_transformation_constraints.append(input_column_modification0)
    attacker.pre_transformation_constraints.append(input_column_modification1)
    return Attack(attacker.goal_function, attacker.constraints + attacker.pre_transformation_constraints,
                  attacker.transformation, attacker.search_method)


def build_weak_attacker(args, model) -> Attack:
    attacker = None
    if args.attack_method == 'textbugger':
        attacker = TextBuggerLi2018.build(model)
    elif args.attack_method == 'textfooler':
        attacker = TextFoolerJin2019.build(model)
    elif args.attack_method == 'bertattack':
        attacker = BAEGarg2019.build(model)
    elif args.attack_method == 'pwws':
        attacker = PWWSRen2019.build(model)
    else:
        print('Not implement attck!')
        exit(41)

    if args.attack_method in ['bertattack']:
        attacker.transformation = WordSwapEmbedding(max_candidates=args.neighbour_vocab_size)
        for constraint in attacker.constraints:
            if isinstance(constraint, WordEmbeddingDistance):
                attacker.constraints.remove(constraint)
            if isinstance(constraint, UniversalSentenceEncoder):
                attacker.constraints.remove(constraint)

    # attacker.constraints.append(MaxWordsPerturbed(max_percent=args.modify_ratio))
    use_constraint = UniversalSentenceEncoder(
        threshold=args.sentence_similarity,
        metric="cosine",
        compare_against_original=True,
        window_size=15,
        skip_text_shorter_than_window=False,
    )
    attacker.constraints.append(use_constraint)
    input_column_modification0= InputColumnModification(["sentence1", "sentence2"], {"sentence1"})
    input_column_modification1 = InputColumnModification(["sentence", "question"], {"sentence"})
    attacker.pre_transformation_constraints.append(input_column_modification0)
    attacker.pre_transformation_constraints.append(input_column_modification1)
    attacker.goal_function = UntargetedClassification(model)
    return Attack(attacker.goal_function, attacker.constraints + attacker.pre_transformation_constraints,
                  attacker.transformation, attacker.search_method)


def attack_parse_args():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument('--do_lower_case', type=bool, default=True)
    parser.add_argument('--model_name_or_path',default='bert-base-uncased',type=str)
    parser.add_argument('--attack_log', default='attack_log.csv', type=str)
    parser.add_argument('--official_log', default='official_log.csv', type=str)
    parser.add_argument('--dataset_name', default='glue', type=str)
    parser.add_argument('--task_name', default=None, type=str)

    parser.add_argument('--out_w_per_mask', type=int, default=1)
    parser.add_argument('--in_w_per_mask', type=float, default=1)
    parser.add_argument('--mask_p', type=float, default=0.9)  # init mask score

    parser.add_argument('--valid', type=str, default='validation')  # test for imdb, agnews; validation for GLUEs
    parser.add_argument('--num_examples', default=1000, type=int)  # number of attack sentences
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--attack_method', type=str, default='textfooler')
    parser.add_argument('--neighbour_vocab_size', default=10, type=int)
    parser.add_argument('--modify_ratio', default=0.15, type=float)
    parser.add_argument('--sentence_similarity', default=0.85, type=float)
    parser.add_argument('--save_perturbed', default=1, type=int)
    parser.add_argument('--perturbed_file', default='results.csv', type=str)
    args = parser.parse_args()
    return args


def main():
    args = attack_parse_args()
    # for model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = MaskedBertForSequenceClassification.from_pretrained(
        args.model_name_or_path, config=config, out_w_per_mask=args.out_w_per_mask,
        in_w_per_mask=args.in_w_per_mask, mask_p=args.mask_p)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model.eval()


    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    if args.attack_method == "bertattack":
        attack = build_weak_attacker(args, model_wrapper)
    else:
        attack = build_default_attacker(args, model_wrapper)

    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news':
        attack_valid = 'test'
    elif args.task_name == 'mnli':
        attack_valid = 'validation_matched'
    else:
        attack_valid = 'validation'

    dataset = HuggingFaceDataset(args.dataset_name, args.task_name, split=attack_valid)

    # for attack
    attack_args = AttackArgs(num_examples=args.num_examples, log_to_csv=args.official_log,
                             disable_stdout=True, random_seed=args.seed)
    attacker = Attacker(attack, dataset, attack_args)
    num_results = 0
    num_successes = 0
    num_failures = 0
    for result in attacker.attack_dataset():
        logger.info(result)
        num_results += 1
        if (
                type(result) == SuccessfulAttackResult
                or type(result) == MaximizedAttackResult
        ):
            num_successes += 1
        if type(result) == FailedAttackResult:
            num_failures += 1

        if args.save_perturbed:
            with open(args.perturbed_file, 'a', encoding='utf-8', newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([result.perturbed_result.attacked_text.text, result.perturbed_result.ground_truth_output])

    logger.info("[Succeeded / Failed / Total] {} / {} / {}".format(num_successes, num_failures, num_results))
    # compute metric
    original_accuracy = (num_successes + num_failures) * 100.0 / num_results
    accuracy_under_attack = num_failures * 100.0 / num_results

    if original_accuracy != 0:
        attack_succ = (original_accuracy - accuracy_under_attack) * 100.0 / original_accuracy
    else:
        attack_succ = 0
    out_csv = open(args.attack_log, 'a', encoding='utf-8', newline="")
    csv_writer = csv.writer(out_csv)
    csv_writer.writerow([args.model_name_or_path, original_accuracy, accuracy_under_attack, attack_succ])
    out_csv.close()


if __name__ == "__main__":
    main()
