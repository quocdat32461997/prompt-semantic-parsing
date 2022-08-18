import argparse
from torchtools.configs import Configs
import pytorch_lightning as pl
from psp.dataset import *

def main(args):
    # Read and parse configs
    configs = Configs(path=args.config_path).get_configs()

    # Creata dataloaders
    train_dataloader = None
    val_dataloader = None
    test_dataloader = None

    # Create model
    model = None

    # Create trainer
    trainer = pl.Trainer.from_argparse_args(configs)

    if args.run_type == 'train':
        trainer.fit(model, train_dataloader, val_dataloader)
    elif args.run_type == 'test':
        trainer.test(test_dataloader)

if __name__ == '__main__':
    # Init parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', type=str, required=True)
    parser.add_argument('--run-type', type=str, required=True)

    main(parser.parse_args())
