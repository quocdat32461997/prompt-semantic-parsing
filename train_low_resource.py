import argparse
from torchtools.configs import Configs
import pytorch_lightning as pl
from psp.constants import PRETRAINED_BART_MODEL, DatasetPaths, RunMode
from psp.models import Seq2SeqCopyPointer, LowResourceSemanticParser
from psp.dataset import Tokenizer, LowResourceTOpv2Dataset, SMPDataLoader


def setup(configs, **kwargs):
    if configs.data == "topv2":
        dataset_path = DatasetPaths.TOPv2
        dataset = LowResourceTOpv2Dataset
    else:
        raise ValueError("{} dataset is not a valid choie.".format(configs.data))
    # Inint tokenizer
    print("Initiating tokenizer.")
    tokenizer = Tokenizer(pretrained=PRETRAINED_BART_MODEL, dataset_path=dataset_path)

    # Creata dataloaders
    print("Initiating data loaders.")
    train_dataloader = SMPDataLoader(
        tokenizer=tokenizer,
        dataset=dataset(bucket=RunMode.TRAIN),
        dataset_path=dataset_path,
        batch_size=configs.batch_size,
        shuffle=True,
    )
    val_dataloader = SMPDataLoader(
        tokenizer=tokenizer,
        dataset=dataset(bucket=RunMode.EVAL),
        dataset_path=dataset_path,
        batch_size=configs.batch_size,
    )
    test_dataloader = SMPDataLoader(
        tokenizer=tokenizer,
        dataset=dataset(bucket=RunMode.TEST),
        dataset_path=dataset_path,
        batch_size=configs.batch_size,
    )

    # Built models
    print("Initiating model: {}.".format(configs.model_name))
    if configs.model_name == "Seq2SeqCopyPointer":
        model = Seq2SeqCopyPointer(
            pretrained=PRETRAINED_BART_MODEL,
            vocab_size=tokenizer.vocab_size,
            ontology_vocab_ids=tokenizer.ontology_vocab_ids,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        raise ValueError("{} model is not a valid choice.".format(configs.model_name))

    print("Initiating parser: {}.".format(configs.parser_name))
    if configs.parser_name == "LowResourceSemanticParser":
        model = LowResourceSemanticParser(model=model, lr=configs.lr)
    else:
        raise ValueError("{} parser is not a valid choice.".format(configs.parser_name))

    return model, {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader,
    }


def main(args):
    # Read and parse configs
    print("Reading configs")
    configs = Configs(path=args.path_to_config)

    # Create model
    model, dataloaders = setup(configs=configs)

    # Create trainer
    print("Initiating trainer.")
    trainer = pl.Trainer.from_argparse_args(configs, default_root_dir=configs.save_dir)

    # Train
    print("Training.")
    trainer.fit(model, dataloaders[RunMode.TRAIN.value])  # , dataloader["val"])
    # Test
    print("Testing.")
    # trainer.test(dataloader["test"])


if __name__ == "__main__":
    # Init parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-config", type=str, required=True)
    main(parser.parse_args())
