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


def visualize_metrics(thresholds, f1, auroc, out):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure()
    ax = plt.axes()

    plt.plot(thresholds, f1, color='blue', label='F1')
    plt.plot(thresholds, auroc, color='red', label='AUROC')
    plt.title("Crack Segmentation Results")
    plt.xlabel("Confidence")
    plt.legend()

    plt.savefig(out)


def validation(model, dataloader, metrics, th, out, f1, auroc):
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
                    batchsummary[f'Validation_{name}'].append(metric(y_true > 0, y_pred > th, zero_division=0))
                else:
                    try:
                        batchsummary[f'Validation_{name}'].append(metric(y_true.astype('uint8'), y_pred))
                    except:
                        pass
    
    time_elapsed = time.time() - start

    for field in fieldnames:
        batchsummary[field] = np.mean(batchsummary[field])

    print('Validation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Confidence threshold: {th}')
    print(f'Validation results:\n{batchsummary}')
    with open(f'{out}/metrics.txt', 'w') as metrics:
        metrics.write(f'Validation results:\n{batchsummary}')
    f1.append(batchsummary['Validation_f1_score'])
    auroc.append(batchsummary['Validation_auroc'])
        


@click.command()
@click.option("--data_dir", required=True, help="Specify the root data directory")
@click.option("--out_dir", required=True, help="Specify the output directory")
@click.option("--name", required=False, help="Name of the experiment")
@click.option("--model_file", required=True, help="Specify the model file (.pt)")
def main(data_dir, out_dir, name, model_file):
    # Load model
    model = torch.load(model_file)
    print(f'Model loaded.')

    data_dir = Path(data_dir)
    out_directory = Path(os.path.join(out_dir, name))
    if not out_directory.exists():
        out_directory.mkdir()

    th = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    dataloader = datahandler.get_dataloader_single_folder_val(data_dir=data_dir, batch_size=1)

    # Specify the evaluation metrics
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    f1_list = []
    auroc_list = []
    for i in th:
        th_folder = Path(os.path.join(out_directory, str(i)))
        if not th_folder.exists():
            th_folder.mkdir()
        validation(model, dataloader, metrics, i, th_folder, f1_list, auroc_list)

    visualize_metrics(th, f1_list, auroc_list, os.path.join(out_dir, name, 'f1_auroc.png'))

if __name__ == "__main__":
    main()