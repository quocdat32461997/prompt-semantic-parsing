import os
import pickle
from collections import defaultdict
from typing import Dict, List, Union
import pandas as pd
from psp.constants import ONTOLOGY_SCOPE_PATTERN, ONTOLOGY_PATTERN, OntologyVocabs, TOPv2_DOMAIN_MAP, Datasets
import re


def get_ontology_from_topv2_dataset():
    # Lambda function to find all ontologies
    def find_ontology(row): return re.findall(ONTOLOGY_PATTERN, row)

    data_path_per_domain: Dict[str, str] = {
        'alarm': 'alarm_train.tsv',
        'event': 'event_train.tsv',
        'messaging': 'messaging_train.tsv',
        'music': 'music_train.tsv',
        'navigation': 'navigation_train.tsv',
        'reminder': 'reminder_train.tsv',
        'weather': 'weather_train.tsv',
        'timer': 'timer_train.tsv',
    }

    intents_per_domain: Dict[str, List[str]] = defaultdict(list)
    slots_per_domain: Dict[str, List[str]] = defaultdict(list)

    for domain, path in data_path_per_domain.items():
        df = pd.read_csv(os.path.join(Datasets.TOPv2, path), sep='\t')

        # Find ontology vocabs and their scopes
        ontology_vocabs = df['semantic_parse'].map(find_ontology).explode().unique()
        
        # Map to intents and slots
        for vocab in ontology_vocabs:
            if vocab.startswith('IN:'):
                intents_per_domain[domain].append(vocab)
            elif vocab.startswith('SL:'):
                slots_per_domain[domain].append(vocab)
            else:
                raise ValueError("{} is not a valid ontology.".format(vocab))

    # Create a comprehensive ontology vocab
    vocabs: Dict[str, Union[int, Dict[str, List[str]]]] = {
        domain: {
            "intents": intents_per_domain[domain],
            "slots": slots_per_domain[domain],
        } for domain in data_path_per_domain.keys()
    }

    # Save vocabs
    with open(OntologyVocabs.TOPv2.value, 'wb') as file:
        pickle.dump(vocabs, file)


def main():
    # Get Ontology vocabs from  TOPv2 dataset
    get_ontology_from_topv2_dataset()


if __name__ == '__main__':
    main()
