# prompt-semantic-parsing
Prompting for Task-Oriented Semantic Parsing
* This repo was inspired by the hierarchical design of several pre-ChatGPT chatbots: identify intent A -> semantic parsing for intent A. The design, IMO, accelerates inference and scalability to adapt new domains (few-shot learning).
* The repo idea was to combine prompting + BART to reduce training effort on new domains. In the ideal settings, new domains require updating corresponding sets of intents-slots only. The decoder will fill the following prompt “Given text, the intent is [IN: X]. The slots are [SL: Y phrase [SL: Z phrase ] ].
  
## Datasets
* SNIPS
* TOP
* TOPv2
* Overnight

## Training

Retrieve ontologies and pointers
```
python3 get_ontology.py
python3 get_poitners.py
```

Start training low_resource experiments
```
python3 train_low_resource.py --path-to-config configs/path_to_training_configs_per_dataset.json
```

## Testing

## Baselines
* [Low-Resource Domain Adaption for Compositional Task-Oriented Semantic Parsing](https://github.com/quocdat32461997/prompt-semantic-parsing/blob/main/readings/low-resource%20task-oriented%20semantic%20parsing.pdf)
* [Don’t Parse, Generate! A Sequence to Sequence Architecture for Task-Oriented Semantic Parsing](https://assets.amazon.science/11/1a/f47268a940bfbbe7cc29a8ba4700/scipub-1042.pdf)
* [T5 and BART Prompt Tuning for Semantic Parsing](https://arxiv.org/pdf/2110.08525.pdf)
