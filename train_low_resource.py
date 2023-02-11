import os
import torch
import argparse
from typing import Dict
from torchtools.configs import Configs
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from psp.constants import PRETRAINED_BART_MODEL, DatasetPaths, RunMode

from psp.models import Seq2SeqVocabCopyPointer, Seq2SeqIndexCopyPointer, LowResourceSemanticParser
from psp.dataset import SemanticParseDataModule


def model_info(model: torch.nn.Module) -> None:
    """Display total and trainable parameters of the given model. """
    total_params = sum(
	param.numel() for param in model.parameters()
)

    trainable_params = sum(
	p.numel() for p in model.parameters() if p.requires_grad
)
    print('Total Params: {} \nTrainable Params: {}'.format(total_params, trainable_params))


def setup(configs, **kwargs):
    # Initiate data-module
    print("Initializing data-module.")
    data_module = SemanticParseDataModule(
        dataset_name=configs.data,
        batch_size=configs.batch_size,
        num_workers=configs.num_workers,
        pretrained=configs.pretrained_tokenizer,
        use_processed_data=configs.use_processed_data,
        use_pointer_data=configs.use_pointer_data,)

    # Built models
    print("Initiating model: {}.".format(configs.model_name))
    if configs.model_name == "Seq2SeqVocabCopyPointer":
        core_model = Seq2SeqVocabCopyPointer(
            pretrained=PRETRAINED_BART_MODEL,
            vocab_size=data_module.transform.tokenizer.vocab_size,
            ontology_vocab_ids=data_module.transform.tokenizer.ontology_vocab_ids,
            bos_token_id=data_module.transform.tokenizer.bos_token_id,
            eos_token_id=data_module.transform.tokenizer.eos_token_id,
            pad_token_id=data_module.transform.tokenizer.pad_token_id,
            beam_size=configs.beam_size,
            alpha=configs.alpha,
            reward=configs.reward,
            max_queue_size=configs.max_queue_size,
            n_best=configs.n_best,
            min_dec_steps=configs.min_dec_steps,
            dropout=configs.dropout,
        )
    elif configs.model_name == "Seq2SeqIndexCopyPointer":
        core_model = Seq2SeqIndexCopyPointer(
            pretrained=PRETRAINED_BART_MODEL,
            vocab_size=data_module.transform.tokenizer.vocab_size,
            output_vocab_size=data_module.transform.tokenizer.output_vocab_size,
            ontology_vocab_ids=data_module.transform.tokenizer.ontology_vocab_ids,
            bos_token_id=data_module.transform.tokenizer.bos_token_id,
            eos_token_id=data_module.transform.tokenizer.eos_token_id,
            pad_token_id=data_module.transform.tokenizer.pad_token_id,
            beam_size=configs.beam_size,
            alpha=configs.alpha,
            reward=configs.reward,
            max_queue_size=configs.max_queue_size,
            n_best=configs.n_best,
            min_dec_steps=configs.min_dec_steps,
            dropout=configs.dropout,
        )
    else:
        raise ValueError("{} model is not a valid choice.".format(configs.model_name))
    
    # Information of mdoel
    model_info(core_model)

    print("Initiating parser: {}.".format(configs.parser_name))
    if configs.parser_name == "LowResourceSemanticParser":
        model = LowResourceSemanticParser(
            model=core_model,
            lr=configs.lr,
            intent_id_list=data_module.transform.tokenizer.intent_id_list,
            slot_id_list=data_module.transform.tokenizer.slot_id_list,
            ontology_id_list=data_module.transform.tokenizer.ontology_vocab_ids,
            vocab_size=data_module.transform.tokenizer.vocab_size,
        )
    else:
        raise ValueError("{} parser is not a valid choice.".format(configs.parser_name))

    return model, data_module


def main(args):
    # Read and parse configs
    print("Reading configs")
    configs: Configs = Configs(path=args.path_to_config)

    # Create model
    model, data_module = setup(configs=configs)

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
        trainer.fit(model, data_module)

    if args.test:
        # Test
        print("Testing.")
        trainer.test(model, data_module)

if __name__ == "__main__":
    # Init parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-to-config", type=str, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    main(parser.parse_args())
