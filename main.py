from pathlib import Path

import click
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils import data

import datahandler
from model import createDeepLabv3
from trainer import train_model


@click.command()
@click.option("--data-directory",
              required=True,
              help="Specify the data directory.")
@click.option("--exp_directory",
              required=True,
              help="Specify the experiment directory.")
@click.option(
    "--epochs",
    default=25,
    type=int,
    help="Specify the number of epochs you want to run the experiment for.")
@click.option("--batch-size",
              default=4,
              type=int,
              help="Specify the batch size for the dataloader.")
@click.option("--model_file", required=False, help="Specify the model file (.pt) or leave empty for default config.")
def main(data_directory, exp_directory, epochs, batch_size, model_file):
    # Create the deeplabv3 resnet101 model
    if model_file is None:
        # Pretrained model on subset of COCO train2017
        model = createDeepLabv3()
    else:
        # Custom weights
        model = torch.load(model_file)
    model.train()
    data_directory = Path(data_directory)
    # Create the experiment directory if not present
    exp_directory = Path(exp_directory)
    if not exp_directory.exists():
        exp_directory.mkdir()

    # Specify the loss function
    criterion = torch.nn.MSELoss(reduction='mean')
    # Specify the optimizer with a lower learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Specify the evaluation metrics
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    # Create the dataloader
    dataloaders = datahandler.get_dataloader_single_folder(
        data_directory, batch_size=batch_size)
    _ = train_model(model,
                    criterion,
                    dataloaders,
                    optimizer,
                    bpath=exp_directory,
                    metrics=metrics,
                    num_epochs=epochs)

    # Save the trained model
    torch.save(model, exp_directory / 'weights.pt')


if __name__ == "__main__":
    main()
