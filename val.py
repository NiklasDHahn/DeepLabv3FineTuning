import os 
import time
import numpy as np
import torch
from tqdm import tqdm
import click
from sklearn.metrics import f1_score, roc_auc_score
import datahandler
from pathlib import Path


def validation(model, dataloader, metrics):
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    fieldnames = [f'Validation_{m}' for m in metrics.keys()]
    batchsummary = {a: [0] for a in fieldnames}

    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        start = time.time()
        for sample in tqdm(iter(dataloader)):
            inputs = sample['image'].to(device)
            masks = sample['masks'].to(device)

            outputs = model(inputs)

            y_pred = outputs['out'].data.cpu().numpy().ravel()
            y_true = masks.data.cpu().numpy().ravel()

            for name, metric in metrics.items():
                if name == 'f1_score':
                    # Use a classification threshold of 0.1
                    batchsummary[f'Validation_{name}'].append(metric(y_true > 0, y_pred > 0.1))
                else:
                    batchsummary[f'Validation_{name}'].append(metric(y_true.astype('uint8'), y_pred))
    
    time_elapsed = time.time() - start

    for field in fieldnames:
        batchsummary[field] = np.mean(batchsummary[field])

    print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Validatin results:\n{batchsummary}')
        


@click.command()
@click.option("--data_dir", required=True, help="Specify the root data directory")
@click.option("--out_dir", required=True, help="Specify the output directory")
@click.option("--model_file", required=True, help="Specify the model file (.pt)")
@click.option("--thresh", default=.5, type=float, help="Set the decision threshold for prediction")
def main(data_dir, out_dir, model_file, thresh):
    # Load model
    model = torch.load(model_file)
    print(f'Model summary:\n {model.eval()}')

    data_dir = Path(data_dir)
    out_directory = Path(out_dir)
    if not out_directory.exists():
        out_directory.mkdir()

    dataloader = datahandler.get_dataloader_single_folder_val(data_dir=data_dir, batch_size=3)

    # Specify the evaluation metrics
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}


if __name__ == "__main__":
    main()