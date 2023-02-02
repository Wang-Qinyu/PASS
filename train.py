import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer
from detectron2.data import MetadataCatalog, DatasetCatalog

from dataset import *

from detectron2.utils.logger import setup_logger


cfg = get_cfg()
# cfg.OUTPUT_DIR = './output/BCCD'
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
cfg.DATASETS.TRAIN = ("circle_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 8
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 16  #batch size
cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR

ITERS_IN_ONE_EPOCH =  2898 #//cfg.SOLVER.IMS_PER_BATCH + 1  #
cfg.SOLVER.MAX_ITER =   20000 #(ITERS_IN_ONE_EPOCH * 2) - 1    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset


cfg.SOLVER.STEPS = [2000,5000,8000]        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (cell).

cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH
# cfg.num-gpus = ITERS_IN_ONE_EPOCH
    
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()