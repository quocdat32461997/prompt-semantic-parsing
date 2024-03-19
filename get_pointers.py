import re
import os
import torch
import pickle
from typing import Dict, List
import pandas as pd
import argparse
from psp.dataset.data_utils import read_top_dataset
from psp.constants import ONTOLOGY_SCOPE_PATTERN, DatasetPaths, PRETRAINED_BART_MODEL, MULTI_WHITESPACE_PATTERN, SINGLE_SPACE
from psp.dataset import PointerTokenizer, Tokenizer

def init_tokenizer(dataset_path: str):
    return PointerTokenizer(pretrained=PRETRAINED_BART_MODEL, dataset_path=dataset_path)

def parse_pointers(tokenizer, df) -> Dict[str, List[int]]:

    parse_seq: List[int] = []
    utter_seq: List[int] = []
    pointer_seq: List[int] = []
    utterance: str = ""
    last: int = 0
    last_seq: int = 0

    matches = re.finditer(ONTOLOGY_SCOPE_PATTERN, df.semantic_parse)
    for match in matches:
        start, end = match.span()
    
        # Update utterance with prepending text
        utterance += " " + df.semantic_parse[last:start]
        utterance = utterance.strip()#re.sub(MULTI_WHITESPACE_PATTERN, SINGLE_SPACE, utterance).strip()

        # Get token-ids
        ontology_token_id = tokenizer(df.semantic_parse[start:end])["input_ids"][1]
        token_ids = tokenizer(utterance)["input_ids"][1:-1]

        # Update utterance seq
        utter_seq.extend(token_ids)

        # Update parse_seq
        parse_seq.extend(token_ids[last_seq:])
        parse_seq.append(ontology_token_id)

        # Update pointer_seq
        pointer_seq.extend([tokenizer("@ptr{}".format(idx))["input_ids"][1] for idx in range(last_seq, len(token_ids))])
        pointer_seq.append(ontology_token_id)

        last_seq = len(token_ids)
        last = end

    # Add <BOS> and <EOS>
    utter_seq = [tokenizer.tokenizer.bos_token_id] + utter_seq + [tokenizer.tokenizer.eos_token_id]
    parse_seq = [tokenizer.tokenizer.bos_token_id] + parse_seq + [tokenizer.tokenizer.eos_token_id]
    pointer_seq = [tokenizer.tokenizer.bos_token_id] + pointer_seq + [tokenizer.tokenizer.eos_token_id]

    return {"domain": df.domain, 
            "utterance": utter_seq, 
            "semantic_parse": parse_seq, 
            "pointer_parse": pointer_seq
            }

def get_pointers_from_top_dataset(tokenizer: Tokenizer) -> None:
    for set in ["train", "eval", "test"]:
        df: pd.DataFrame = read_top_dataset(
                os.path.join(DatasetPaths.TOP.value, set + ".tsv")
        )
            
        # Generate pointer_parse
        processed_data = df.apply(lambda x: parse_pointers(tokenizer, x), axis=1)

        # Save data
        processed_data = pd.DataFrame.from_dict(processed_data)
        processed_data.to_csv(os.path.join(DatasetPaths.TOP.value, "processed_{}.tsv".format(set)))

def get_pointers_from_topv2_dataset(tokenizer: Tokenizer) -> None:
    for set in ["train", "eval", "test"]:
        print("Processing {} set".format(set))
        processed_data = []
        for domain in ["alarm", "event", "messaging", "music", "navigation", "reminder", "weather", "timer"]:    
            path = domain + "_" + set + ".tsv"
            df: pd.DataFrame = pd.read_csv(
                os.path.join(DatasetPaths.TOPv2.value, path), sep="\t",
            )
            
            # Generate pointer_parse
            processed_data.extend(df.apply(lambda x: parse_pointers(tokenizer, x), axis=1))

        # Save data
        with open(os.path.join(DatasetPaths.TOPv2.value, "processed_{}.pkl".format(set)), 'wb') as file:
            pickle.dump(processed_data, file)
        #processed_data = pd.DataFrame.from_dict(processed_data)
        #processed_data.to_csv(os.path.join(DatasetPaths.TOPv2.value, "processed_{}.tsv".format(set)))
        
def main(args):
    # Get Ontology vocabs from  atasets: TOP, TOPV2
    if args.dataset == "top":
        data_path = DatasetPaths.TOP
        func = get_pointers_from_top_dataset
    elif args.dataset == "topv2":
        data_path = DatasetPaths.TOPv2
        func = get_pointers_from_topv2_dataset
    else:
        raise ValueError("{} is a not valid choice.".format(args.dataset))

    # Init tokenizer
    print("Initializing tokenizer")
    tokenizer = init_tokenizer(data_path)

    # Process dataset
    print("Processing data")
    func(tokenizer)

    # Save tokenizer
    print("Saving tokenizer.")
    data_path = os.path.join(data_path.value, "tokenizer")
    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    tokenizer.save_pretrained(data_path)

if __name__ == "__main__":
    # Init parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    main(parser.parse_args())