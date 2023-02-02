from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances

import os
import pickle
from PIL import Image
import random
import cv2


def get_circle_dicts_fixed(img_dir,label_dir="../dataset/GLabels"):
    image_path = img_dir
    samples = os.listdir(image_path)
    
    dataset_dicts = []
    
    for idx, img_name in enumerate(samples):
        record = {}
        filename = os.path.join(image_path,img_name)
        
        image = Image.open(filename)
        w, h = image.size
        
        record['file_name'] = filename
        record['image_id'] = idx 
        record['height'] = h 
        record['width'] = w 
        
        label_path = os.path.join(label_dir,'{}.pkl'.format(img_name.split('.')[0]))
        with open(label_path,'rb') as fp:
            gt = pickle.load(fp)
        
        objs = []
        for anno in gt:
            assert len(anno) == 4, 'length error!'
            xmin,ymin,xmax,ymax = anno
            poly = [ (xmin, ymin),(xmax, ymin),(xmax, ymax),(xmin, ymax),(xmin, ymin) ]
            obj = {
                'bbox': anno,
                'bbox_mode': BoxMode.XYXY_ABS,
                'segmentation': poly,
                'category_id':0,
            }
            objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)
    
    #! add real images   dataset_dicts
    image_path = '../dataset/images/train_real'
    label_dir = '../dataset/labels'
    samples = os.listdir(image_path)
    for _ in range(20):
        dataset_real = []
        for idx, img_name in enumerate(samples):
            record = {}
            filename = os.path.join(image_path,img_name)
            
            image = Image.open(filename)
            w, h = image.size
            
            record['file_name'] = filename
            record['image_id'] = idx 
            record['height'] = h 
            record['width'] = w 
            
            label_path = os.path.join(label_dir,'{}.pkl'.format(img_name.split('.')[0]))
            with open(label_path,'rb') as fp:
                gt = pickle.load(fp)
            
            objs = []
            for anno in gt:
                assert len(anno) == 4, 'length error!'
                xmin,ymin,xmax,ymax = anno
                poly = [ (xmin, ymin),(xmax, ymin),(xmax, ymax),(xmin, ymax),(xmin, ymin) ]
                obj = {
                    'bbox': anno,
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'segmentation': poly,
                    'category_id':0,
                }
                objs.append(obj)
            record['annotations'] = objs
            dataset_real.append(record)
        dataset_dicts.extend(dataset_real)
    
    return dataset_dicts

def get_circle_dicts(img_dir,label_dir="../dataset/labels"):
    image_path = img_dir
    samples = os.listdir(image_path)
    
    dataset_dicts = []
    
    for idx, img_name in enumerate(samples):
        record = {}
        filename = os.path.join(image_path,img_name)
        
        image = Image.open(filename)
        w, h = image.size
        
        record['file_name'] = filename
        record['image_id'] = idx 
        record['height'] = h 
        record['width'] = w 
        
        label_path = os.path.join(label_dir,'{}.pkl'.format(img_name.split('.')[0]))
        with open(label_path,'rb') as fp:
            gt = pickle.load(fp)
        
        objs = []
        for anno in gt:
            assert len(anno) == 4, 'length error!'
            xmin,ymin,xmax,ymax = anno
            # xmin -= 0.5
            # xmax += 0.5
            # ymin -= 0.5 
            # ymax += 0.5
            poly = [ (xmin, ymin),(xmax, ymin),(xmax, ymax),(xmin, ymax),(xmin, ymin) ]
            obj = {
                'bbox': anno,
                'bbox_mode': BoxMode.XYXY_ABS,
                'segmentation': poly,
                'category_id':0,
            }
            objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ['train','test']:
    DatasetCatalog.register("circle_" + d, lambda d=d: get_circle_dicts(f"../dataset/images/{d}"))
    MetadataCatalog.get("circle_" + d).set(thing_classes=["circle"])

#! for train:
# for d in ['train']:
#     DatasetCatalog.register("circle_" + d, lambda d=d: get_circle_dicts_fixed(f"../dataset/images/{d}"))
#     MetadataCatalog.get("circle_" + d).set(thing_classes=["circle"])
# for d in ['train','test']:
#     DatasetCatalog.register("circle_" + d, lambda d=d: get_circle_dicts("../dataset/pre-masks",'../dataset/labels'))
#     MetadataCatalog.get("circle_" + d).set(thing_classes=["circle"])

#! NEW
# for d in ['train','test']:
#     if d == 'train':
#         DatasetCatalog.register("circle_" + d, lambda d=d: get_circle_dicts(f"../dataset/images/{d}"))
#         MetadataCatalog.get("circle_" + d).set(thing_classes=["circle"])
#     else:
#         DatasetCatalog.register("circle_" + d, lambda d=d: get_circle_dicts("../dataset/pre-masks",'../dataset/labels'))
#         MetadataCatalog.get("circle_" + d).set(thing_classes=["circle"])


circle_metadata = MetadataCatalog.get("circle_train") 
dataset_dicts = get_circle_dicts("../dataset/images/train")

# for d in random.sample(dataset_dicts, 380):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=circle_metadata, scale=0.5)
#     out = visualizer.draw_dataset_dict(d)
    
#     cv2.imwrite('./demo.png',out.get_image()[:, :, ::-1])