import os
import argparse
from torchtools.configs import Configs
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from psp.constants import PRETRAINED_BART_MODEL, DatasetPaths, RunMode
from psp.models import Seq2SeqCopyPointer, LowResourceSemanticParser
from psp.dataset import (
    Tokenizer,
    LowResourceTOPv2Dataset,
    SMPDataLoader,
    LowResourceTOPDataset,
)


def setup(configs, **kwargs):
    if configs.data == "topv2":
        dataset_path = DatasetPaths.TOPv2
        dataset = LowResourceTOPv2Dataset
    elif configs.data == "top":
        dataset_path = DatasetPaths.TOP
        dataset = LowResourceTOPDataset
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
        run_mode=RunMode.EVAL,
    )
    test_dataloader = SMPDataLoader(
        tokenizer=tokenizer,
        dataset=dataset(bucket=RunMode.TEST),
        dataset_path=dataset_path,
        batch_size=configs.batch_size,
        run_mode=RunMode.EVAL,
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
            beam_size=configs.beam_size,
            alpha=configs.alpha,
            reward=configs.reward,
            max_queue_size=configs.max_queue_size,
            n_best=configs.n_best,
            min_dec_steps=configs.min_dec_steps,
        )
    else:
        raise ValueError("{} model is not a valid choice.".format(configs.model_name))

    print("Initiating parser: {}.".format(configs.parser_name))
    if configs.parser_name == "LowResourceSemanticParser":
        model = LowResourceSemanticParser(
            model=model,
            lr=configs.lr,
            intent_id_list=tokenizer.intent_id_list,
            slot_id_list=tokenizer.slot_id_list,
            ontology_id_list=tokenizer.ontology_vocab_ids,
            vocab_size=tokenizer.vocab_size,
        )
    else:
        raise ValueError("{} parser is not a valid choice.".format(configs.parser_name))

    return model, {
        "train": train_dataloader,
        "eval": val_dataloader,
        "test": test_dataloader,
    }


def main(args):
    # Read and parse configs
    print("Reading configs")
    configs: Configs = Configs(path=args.path_to_config)

    # Create model
    model, dataloaders = setup(configs=configs)

    # Configure experiment  name
    experiment_name = configs.parser_name + "v{}".format(
        1 + len(os.listdir(configs.save_dir))
    )
    save_dir = configs.save_dir + "/{}".format(experiment_name)

    # Configure callbacks
    early_stopping = EarlyStopping(
        monitor="em_acc", patience=5, verbose=True, mode="max"
    )
    checkpoint = ModelCheckpoint(
        dirpath=save_dir,
        monitor="em_acc",
        verbose=True,
        mode="max",
        every_n_train_steps=configs.checkpoint_steps,
    )
    callbacks = [early_stopping, checkpoint]

    # Configure
    logger = WandbLogger(
        # name=experiment_name,
        project="prompt-semantic-parsing",
        log_model="all",
    )

    # Create trainer
    print("Initiating trainer.")
    trainer = pl.Trainer.from_argparse_args(
        configs,
        enable_progress_bar=True,
        callbacks=callbacks,
        logger=logger,
    )

    if args.train:
        # Train
        print("Training.")
        trainer.fit(
            model,
            train_dataloaders=dataloaders[RunMode.TRAIN.value],
            val_dataloaders=dataloaders[RunMode.EVAL.value],
        )

    if args.test:
        # Test
        print("Testing.")
        trainer.test(model, dataloaders=dataloaders[RunMode.TEST.value])


if __name__ == "__main__":
    # Init parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-config", type=str, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    main(parser.parse_args())
