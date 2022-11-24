import os
import cv2
from absl import app, flags
from glob import glob
import threading
import numpy as np
import math

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '', 'Directory containing datasets.')
flags.DEFINE_integer('workers', 3, 'Number of workers.')
flags.DEFINE_string('file_type', 'jpg', 'File type to convert.')
flags.DEFINE_string('algorithm', 'padding', 'Choose algorithm (padding, size).')


def get_img_width_height(dir: list) -> tuple:
    widths = []
    heights = []

    for img_path in dir:
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        widths.append(w)
        heights.append(h)

    return widths, heights


def add_padding(dir: list, width: int, height: int):
    pad = np.zeros((height, width, 3), dtype='uint8')
    for img_path in dir:
        print(f'Processing {img_path}')
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        lower = int((height - h) / 2) + math.ceil((height - h) / 2 % 2)
        upper = h + int((height - h) / 2) + math.ceil((height - h) / 2 % 2)
        pad[lower:upper, :, :] = img
        cv2.imwrite(img_path, pad)


def reduce_size(dir:list):
    scale_factor = 0.75
    for img_path in dir:
        print(f'Processing {img_path}')
        img = cv2.imread(img_path)
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        dim = (width, height)

        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        cv2.imwrite(img_path, resized)



def worker_split(dataset: list, workers: int) -> list:
    datasets_list = []
    if workers == 1:
        datasets_list.append(dataset)
    elif len(dataset) / workers % 2 == 0:
        split_idx = int(len(dataset) / workers)
        for w in range(workers):
            datasets_list.append(dataset[split_idx * w: split_idx * (w + 1)])
    else:
        overrun = math.ceil(len(dataset) / workers % 2)
        split_idx = split_idx = int(len(dataset) / workers)
        for w in range(workers):
            datasets_list.append(dataset[split_idx * w: split_idx * (w + 1)])
        new_list = datasets_list[-1] + dataset[-overrun:]
        datasets_list.pop(-1)
        datasets_list.append(new_list)
        
    return datasets_list


def main(argv):
    file_list  = glob(f'{FLAGS.data_dir}*.{FLAGS.file_type}')

    widths, heights = get_img_width_height(file_list)

    min_width, max_width, min_height, max_height  = min(widths), max(widths), min(heights), max(heights)

    w_list = worker_split(file_list, FLAGS.workers)

    if FLAGS.algorithm == 'padding':
        threads = list()
        for w in range(FLAGS.workers):
            thread = threading.Thread(target=add_padding, args=(w_list[w], max_width, max_height,))
            threads.append(thread)
            thread.start()

        for index, thread in enumerate(threads):
            thread.join()
    elif FLAGS.algorithm == 'size':
        threads = list()
        for w in range(FLAGS.workers):
            thread = threading.Thread(target=reduce_size, args=(w_list[w],))
            threads.append(thread)
            thread.start()

        for index, thread in enumerate(threads):
            thread.join()


if __name__ == '__main__':
    app.run(main)