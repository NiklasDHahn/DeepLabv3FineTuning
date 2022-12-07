@echo off

call conda activate deeplabv3

call python val.py --data_dir D:\cracks\validation_dataset --out_dir C:\Users\nik\DeepLabv3FineTuning\val_results --model_file "C:\Users\nik\DeepLabv3FineTuning\EETExp\weights.pt" --thresh 0.1