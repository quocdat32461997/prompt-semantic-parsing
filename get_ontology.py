import re
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Union
import pandas as pd
import argparse

from psp.dataset.data_utils import read_top_dataset
from psp.constants import ONTOLOGY_SCOPE_PATTERN, OntologyVocabs, DatasetPaths


def find_ontology(row):
    # Lambda function to find all ontologies
    return re.findall(ONTOLOGY_SCOPE_PATTERN, row)


def get_ontology_from_top_dataset():
    intents, slots = [], []

    df: pd.DataFrame = read_top_dataset(
        os.path.join(DatasetPaths.TOP.value, "train.tsv")
    )

    # Find ontology vocabs and their scopes
    ontology_vocabs = df["semantic_parse"].map(find_ontology).explode().unique()

    # Map to intents and slots
    for vocab in ontology_vocabs:
        if vocab.startswith("[IN:"):
            intents.append(vocab)
        elif vocab.startswith("[SL:"):
            slots.append(vocab)
        elif vocab == "]":
            pass
        else:
            raise ValueError("{} is not a valid ontology.".format(vocab))

    # Create a comprehensive ontology vocab
    vocabs: Dict[str, List[str]] = {
        "intents": list(set(intents)),
        "slots": list(set(slots)),
    }

    # Save vocabs
    with open(OntologyVocabs.TOP.value, "wb") as file:
        pickle.dump(vocabs, file)


def get_ontology_from_topv2_dataset():

    data_path_per_domain: Dict[str, str] = {
        "alarm": "alarm_train.tsv",
        "event": "event_train.tsv",
        "messaging": "messaging_train.tsv",
        "music": "music_train.tsv",
        "navigation": "navigation_train.tsv",
        "reminder": "reminder_train.tsv",
        "weather": "weather_train.tsv",
        "timer": "timer_train.tsv",
    }

    intents_per_domain: Dict[str, List[str]] = defaultdict(list)
    slots_per_domain: Dict[str, List[str]] = defaultdict(list)

    for domain, path in data_path_per_domain.items():
        df = pd.read_csv(os.path.join(DatasetPaths.TOPv2.value, path), sep="\t")

        # Find ontology vocabs and their scopes
        ontology_vocabs = df["semantic_parse"].map(find_ontology).explode().unique()

        # Map to intents and slots
        for vocab in ontology_vocabs:
            if vocab.startswith("[IN:"):
                intents_per_domain[domain].append(vocab)
            elif vocab.startswith("[SL:"):
                slots_per_domain[domain].append(vocab)
            elif vocab == "]":
                pass
            else:
                raise ValueError("{} is not a valid ontology.".format(vocab))

    # Create a comprehensive ontology vocab
    vocabs: Dict[str, Union[int, Dict[str, List[str]]]] = {
        domain: {
            "intents": list(set(intents_per_domain[domain])),
            "slots": list(set(slots_per_domain[domain])),
        }
        for domain in data_path_per_domain.keys()
    }

    # Save vocabs
    with open(OntologyVocabs.TOPv2.value, "wb") as file:
        pickle.dump(vocabs, file)


def main(args):
    # Get Ontology vocabs from  atasets: TOP, TOPV2
    if args.dataset == "top":
        get_ontology_from_top_dataset()
    elif args.dataset == "topv2":
        get_ontology_from_topv2_dataset()
    else:
        raise ValueError("{} is a not valid choice.".format(args.dataset))


if __name__ == "__main__":
    # Init parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    main(parser.parse_args())
