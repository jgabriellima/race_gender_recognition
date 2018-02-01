import tensorflow as tf
# from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
# from tensorflow.python.platform import tf_logging as logging
# from inception_resnet_v1 import inception_resnet_v1, inception_resnet_v1_arg_scope

from multitask_model import losses, read_and_decode, build_model
from make_features import load_image

import os
import sys
import copy
import random

import tensorflow as tf
import numpy as np
import cv2

import face_detector
from face_aligner import *

model_dir='./log/model_0/'
model_graph = model_dir + 'model_iters_final.meta'
checkpoint_path = model_dir + 'model_iters_final'

image_path = './samples/zhuyan.jpg'

detector_path = './model/detectors/'

image_size = 200
minsize = 20
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709 # scale factor



def run(image_path):
    image = load_image(image_path)
    image_aligned = align_face(image)

    `

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print('python predictor.py file_path_to_image')

    else:
        image_path = sys.argv[1]
        run(image_path)


