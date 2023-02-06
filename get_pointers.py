import re
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Union
import pandas as pd
import argparse

from psp.dataset.data_utils import read_top_dataset
from psp.constants import ONTOLOGY_SCOPE_PATTERN, OntologyVocabs, DatasetPaths, PRETRAINED_BART_MODEL
from psp.dataset import PointerTokenizer

def init_tokenizer(dataset_path: str):
    return PointerTokenizer(pretrained=PRETRAINED_BART_MODEL, dataset_path=dataset_path)

def _parse_pointers(utter_seq: List[int], parse_seq: List[int], ontology_vocabs: List[int]) -> List[int]:
    
    utter_idx, parse_idx = 0, 0

    while parse_idx < len(parse_seq) and utter_idx < len(utter_seq):
        # if the current parse-token is one of ontologies, skip
        if parse_seq[parse_idx] in ontology_vocabs:
            parse_idx += 1
            continue 

        # find the next sequence of non-ontology tokens in parse-seq
        end_parse_idx = parse_idx + 1
        while end_parse_idx < len(parse_seq) and parse_seq[end_parse_idx] not in ontology_vocabs:
            end_parse_idx += 1

        # find the first match of parse-seq in utter-seq
        while utter_idx < len(utter_seq) and utter_seq[utter_idx : utter_idx + end_parse_idx - parse_idx] != parse_seq[parse_idx:end_parse_idx]:
            utter_idx += 1

        # record matches
        if utter_seq[utter_idx : utter_idx + end_parse_idx - parse_idx] != parse_seq[parse_idx:end_parse_idx]:
            while parse_idx < end_parse_idx:
                parse_seq[parse_idx] = None # assign @ptr#

    return parse_seq

def parse_pointers(tokenizer, df):
    # Get pointeres
    pointer_seq_list: List[List[int]] = []
    for utter, parse in zip(df['utterance'], df['semantic_parse']):
        utter_tokens = tokenizer(utter, truncation=True, add_special_tokens=True)["input_ids"]
        parse_tokens = tokenizer(parse, truncation=True, add_special_tokens=True)["input_ids"]
            
        # Generate pointers
        pointer_seq_list.append(_parse_pointers(utter_tokens, parse_tokens, ontology_vocabs=tokenizer.ontology_vocab_ids))
    return pointer_seq_list

def get_pointers_from_top_dataset(tokenizer):
    for path in ["train.tsv", "eval.tsv", "test.tsv"]:
        df: pd.DataFrame = read_top_dataset(
            os.path.join(DatasetPaths.TOP.value, path)
        )
        
        # Generate pointer_parse
        df['pointer_parse'] = parse_pointers(tokenizer=tokenizer, df=df)

        # Save df
        df.to_csv(os.path.join(DatasetPaths.TOP.value, "processed_" + path))

def get_pointers_from_topv2_dataset(tokenizer):
    for set in ["train.tsv", "eval.tsv", "test.tsv"]:
        for domain in ["alarm", "event", "messaging", "music", "navigation", "reminder", "weather", "timer"]:    
            path = domain + "_" + set
            df: pd.DataFrame = read_top_dataset(
                os.path.join(DatasetPaths.TOPv2.value, path)
            )
            
            # Generate pointer_parse
            df['pointer_parse'] = parse_pointers(tokenizer=tokenizer, df=df)

            # Save df
            df.to_csv(os.path.join(DatasetPaths.TOP.value, "processed_" + path))

def main(args):
    # Get Ontology vocabs from  atasets: TOP, TOPV2
    if args.dataset == "top":
        tokenizer = init_tokenizer(DatasetPaths.TOP)
        get_pointers_from_top_dataset(tokenizer)
    elif args.dataset == "topv2":
        tokenizer = init_tokenizer(DatasetPaths.TOPv2)
        get_pointers_from_topv2_dataset(tokenizer)
    else:
        raise ValueError("{} is a not valid choice.".format(args.dataset))

if __name__ == "__main__":
    # Init parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    main(parser.parse_args())