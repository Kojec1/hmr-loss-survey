import json
import sys
import time
from glob import glob

print(sys.executable, sys.version)

import numpy as np
import pandas as pd
import tensorflow as tf

from os.path import join, abspath

# for local import
sys.path.append(abspath('..'))

from main.model import Model
from main.config import Config
from main.dataset import Dataset
from main.discriminator import Discriminator
from main.generator import Generator
from main.model_util import batch_align_by_pelvis, batch_compute_similarity_transform, batch_rodrigues
from visualise.trimesh_renderer import TrimeshRenderer
from visualise.vis_util import preprocess_image, visualize, visualize_full

mapping = {
    'mpii_3d': 'mpi',
    'h36m_filtered': 'h36m',
    'h36m': 'h36m',
    'total_cap': 'TC'
}


if __name__ == '__main__':

    with open('vis_config.json') as f:
        vis_config = json.load(f)

    for setting, models in vis_config.items():
        for model in models:
            config = model['config']

            class VisConfig(Config):
                ENCODER_ONLY = True
                LOG_DIR = join('/', 'development', 'datasets', 'logs', setting, config['model'])
                DATA_DIR = join('/', 'development', 'datasets', config['data_dir'])
                DATASETS = config['datasets']
                JOINT_TYPE = config['joint_type']
                INITIALIZE_CUSTOM_REGRESSOR = config['init_custom_regressor']
                BATCH_SIZE = 5

            config = VisConfig()
            model = Model(display_config=False)

            with tf.device('/CPU:0'):
                dataset = Dataset()
                ds_inference = dataset.get_infer()

            # cnt = 0
            for images, kp2d, kp3d, has3d, seq in ds_inference.take(1):
                # cnt += 1
                # if cnt != 100:
                #     continue

                result = model.detect(images)

                images = (images.numpy() + 1.0) / 2.0
                kp2d = kp2d.numpy()
                kp3d = batch_align_by_pelvis(kp3d)
                kp3d = kp3d.numpy()

                cam = result['cam'].numpy()
                vertices = result['vertices'].numpy()
                joints_2D = (result['kp2d'].numpy() + 1.0) * 112.0
                joints_3D = result['kp3d']
                joints_3D = batch_align_by_pelvis(joints_3D).numpy()

                renderer = TrimeshRenderer()
                visualize_full(renderer, images, cam, kp2d, kp3d, joints_2D, joints_3D, vertices)
