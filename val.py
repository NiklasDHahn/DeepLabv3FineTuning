import os 
import time
import numpy as np
import torch
from tqdm import tqdm
import click
from sklearn.metrics import f1_score, roc_auc_score
import datahandler
import matplotlib.pyplot as plt
from pathlib import Path
from random import randrange


def visualization(out_dir, image, gt, pred, idx):
    # Plot the input image, ground truth and the predicted output
    plt.figure(figsize=(30,30));
    plt.subplot(131);
    plt.imshow(image);
    plt.title('Image')
    plt.axis('off');
    plt.subplot(132);
    plt.imshow(gt);
    plt.title('Ground Truth')
    plt.axis('off');
    plt.subplot(133);
    plt.imshow(pred);
    plt.title('Segmentation Output')
    plt.axis('off');
    plt.savefig(f'{out_dir}/SegmentationOutput{idx}.png',bbox_inches='tight')


def validation(model, dataloader, metrics, th, out):
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    fieldnames = [f'Validation_{m}' for m in metrics.keys()]
    batchsummary = {a: [0] for a in fieldnames}

    # Get random integer to draw sample for output
    rnd_idx = [randrange(len(dataloader)) for i in range(16)]

    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        start = time.time()
        cnt = 0
        for sample in tqdm(iter(dataloader)):
            inputs = sample['image'].to(device)
            masks = sample['mask'].to(device)

            outputs = model(inputs)

            y_pred = outputs['out'].data.cpu().numpy().ravel()
            y_true = masks.data.cpu().numpy().ravel()
            if cnt in rnd_idx:
                visualization(out, inputs[0].cpu().numpy().transpose(1,2,0), 
                            masks[0].cpu().numpy().transpose(1,2,0), 
                            outputs['out'].cpu().detach().numpy()[0][0] > th,
                            cnt)
            cnt += 1

            for name, metric in metrics.items():
                if name == 'f1_score':
                    # Use a classification threshold of 0.1
                    batchsummary[f'Validation_{name}'].append(metric(y_true > 0, y_pred > th))
                else:
                    batchsummary[f'Validation_{name}'].append(metric(y_true.astype('uint8'), y_pred))
    
    time_elapsed = time.time() - start

    for field in fieldnames:
        batchsummary[field] = np.mean(batchsummary[field])

    print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Validation results:\n{batchsummary}')
    with open(f'{out}/matrics.txt', 'w') as metrics:
        metrics.write(f'Validation results:\n{batchsummary}')
        


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

    dataloader = datahandler.get_dataloader_single_folder_val(data_dir=data_dir, batch_size=1)

    # Specify the evaluation metrics
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    validation(model, dataloader, metrics, thresh, out_directory)


if __name__ == "__main__":
    main()