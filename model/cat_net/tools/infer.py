import sys, os
path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import pprint
import shutil

import logging
import time
import timeit
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F

from lib import models
from lib.config import config
from lib.config import update_config
from lib.core.criterion import CrossEntropy, OhemCrossEntropy
from lib.core.function import train, validate
from lib.utils.modelsummary import get_model_summary
from lib.utils.utils import create_logger, FullModel, get_rank

from Splicing.data.data_core import SplicingDataset as splicing_dataset
from pathlib import Path
from project_config import dataset_paths
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import cv2
from lib.models.network_CAT import get_seg_model
def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def main():
    # args = parse_args()
    # Instead of using argparse, force these args:

    ## CHOOSE ##
    args = argparse.Namespace(cfg='experiments/CAT_full.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_full/CAT_full_v2.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    # args = argparse.Namespace(cfg='experiments/CAT_DCT_only.yaml', opts=['TEST.MODEL_FILE', 'output/splicing_dataset/CAT_DCT_only/DCT_only_v2.pth.tar', 'TEST.FLIP_TEST', 'False', 'TEST.NUM_SAMPLES', '0'])
    update_config(config, args)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    ## CHOOSE ##
    test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('RGB', 'DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)  # full model
    # test_dataset = splicing_dataset(crop_size=None, grid_crop=True, blocks=('DCTvol', 'qtable'), DCT_channels=1, mode='arbitrary', read_from_jpeg=True)  # DCT stream

    print(test_dataset.get_info())

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # must be 1 to handle arbitrary input sizes
        shuffle=False,  # must be False to get accurate filename
        num_workers=1,
        pin_memory=False)

    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=test_dataset.class_weights).cuda()
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                 weight=test_dataset.class_weights).cuda()

    model = eval('models.' +config.MODEL.NAME +
                 '.get_seg_model')(config)
    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        raise ValueError("Model file is not specified.")
    model_state_file = "CAT_full_v2.pth.tar"
    print('=> loading model from {}'.format(model_state_file))
    model = FullModel(model, criterion)
    checkpoint = torch.load(model_state_file)
    model.model.load_state_dict(checkpoint['state_dict'])
    print("Epoch: {}".format(checkpoint['epoch']))
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    dataset_paths['SAVE_PRED'].mkdir(parents=True, exist_ok=True)


    def get_next_filename(i):
        dataset_list = test_dataset.dataset_list
        it = 0
        while True:
            if i >= len(dataset_list[it]):
                i -= len(dataset_list[it])
                it += 1
                continue
            name = dataset_list[it].get_tamp_name(i)
            name = os.path.split(name)[-1]
            return name

    with torch.no_grad():
        for index, (image, label, qtable) in enumerate(tqdm(testloader)):
            size = label.size()
            image = image
            # print(type(image))
            label = label.long()
            model.eval()
            _, pred = model(image, label, qtable)
            pred = torch.squeeze(pred, 0)
            pred = F.softmax(pred, dim=0)[1]
            pred = pred.cpu().numpy()

            # filename
            filename = os.path.splitext(get_next_filename(index))[0] + ".png"
            filepath = dataset_paths['SAVE_PRED']/ filename

            # plot
            try:
                width = pred.shape[1]  # in pixels
                fig = plt.figure(frameon=False)
                dpi = 40  # fig.dpi
                fig.set_size_inches(width / dpi, ((width * pred.shape[0])/pred.shape[1]) / dpi)
                plt.imshow(pred, cmap='gray', vmin=0, vmax=1)
                plt.axis('off')
                plt.savefig(filepath, bbox_inches='tight', transparent=True, pad_inches=0)
                plt.close(fig)

                # input_dir = Path("input")
                # for image_path in input_dir.glob("*.[jp][pn]g"):  # matches jpg, jpeg, png files
                image = cv2.imread("input\sample_7.png")
                # Use the function to find bounding boxes
                mask = cv2.imread(str(filepath))
                # print(mask.shape)
                # print(image.shape)
                # resize original image to match mask
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                print(mask.shape)
                print(image.shape)

                # binarise the mask
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

                # load the mask and create rectangle bounding boxes
                min_mask_value = 0.02 * 256
                _, filtered_mask = cv2.threshold(mask, min_mask_value, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_area_threshold = 0.01 * mask.shape[0] * mask.shape[1]
                max_area_threshold = 0.1 * mask.shape[0] * mask.shape[1]
                valid_contours = []
                # image = cv2.imread("input/sample_9.jpg")
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > min_area_threshold and area < max_area_threshold:
                        valid_contours.append(contour)

                # Remove boxes that are inside another box
                def is_inside(inner, outer):
                    x_inner, y_inner, w_inner, h_inner = cv2.boundingRect(inner)
                    x_outer, y_outer, w_outer, h_outer = cv2.boundingRect(outer)
                    return (x_inner >= x_outer and y_inner >= y_outer and
                            x_inner + w_inner <= x_outer + w_outer and
                            y_inner + h_inner <= y_outer + h_outer)

                filtered_contours = []
                for i, contour in enumerate(valid_contours):
                    if not any(is_inside(contour, other) for j, other in enumerate(valid_contours) if i != j):
                        filtered_contours.append(contour)

                for contour in filtered_contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

                cv2.imwrite(str(filepath).replace(".png", f"bbox.jpg"), image)
            
            except:
                print(f"Error occurred while saving output. ({get_next_filename(index)})")

if __name__ == '__main__':
    main()