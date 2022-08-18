import argparse
from torchtools.configs import Configs
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from psp.constants import PRETRAINED_BART_MODEL, Datasets
from psp.models import Seq2SeqCopyPointer
from psp.dataset import Tokenizer, LowResourceTOpv2Dataset


def main(args):
    # Read and parse configs
    print("Reading configs")
    configs = Configs(path=args.config_path)
    
    # Inint tokenizer
    print("Initiating tokenizer.")
    tokenizer = Tokenizer(pretrained=PRETRAINED_BART_MODEL, dataset=Datasets.TOPv2)

    # Creata dataloaders
    print("Initiating data loaders.")
    train_dataloader = DataLoader(dataset=LowResourceTOpv2Dataset(tokenizer=tokenizer, bucket='train'),
                                  batch_size=configs.batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=LowResourceTOpv2Dataset(tokenizer=tokenizer, bucket='eval'),
                                batch_size=configs.batch_size)
    test_dataloader = DataLoader(dataset=LowResourceTOpv2Dataset(
        tokenizer=tokenizer, bucket='test'), batch_size=configs.batch_size)

    # Create model
    print("Initiating Seq2SeqCopyPointer.")
    model = Seq2SeqCopyPointer(pretrained=PRETRAINED_BART_MODEL, ontology_vocab_size=tokenizer.ontology_vocab_size, bos_token_id=tokenizer.bos_token_id,
                               eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)

    # Create trainer
    print("Initiating trainer.")
    trainer = pl.Trainer.from_argparse_args(configs)

    # Train
    print("Training.")
    #trainer.fit(model, train_dataloader, val_dataloader)

    # Test
    print("Testing.")
    #trainer.test(test_dataloader)


if __name__ == '__main__':
    # Init parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', type=str, required=True)

    main(parser.parse_args())
