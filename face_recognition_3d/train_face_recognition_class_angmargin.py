'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
from __future__ import print_function

import argparse
import math
from datetime import datetime
import time
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
import ast
THIS_FILE_NAME = os.path.basename(__file__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, '..'))
sys.path.append(os.path.join(ROOT_DIR, '../models'))
sys.path.append(os.path.join(ROOT_DIR, '../utils'))
import provider
import tf_util
from pc_util import write_obj

# Bernardo
from plots import plots_fr_pointnet2

# import modelnet_dataset     # original
# import modelnet_h5_dataset  # original
from data_loader.loader_frgc2 import frgc2_dataset                                         # Bernardo
from data_loader.loader_synthetic_faces_gpmm import synthetic_faces_gpmm_dataset           # Bernardo
from data_loader.loader_reconstructed_MICA import lfw_3Dreconstructed_MICA_dataset         # Bernardo
from data_loader.loader_reconstructed_MICA import ms1mv2_3Dreconstructed_MICA_dataset      # Bernardo
from data_loader.loader_reconstructed_MICA import webface_3Dreconstructed_MICA_dataset     # Bernardo
from data_loader.loader_reconstructed_MICA import casia_3Dreconstructed_MICA_dataset       # Bernardo
from data_loader.loader_reconstructed_MICA import ffhq_3Dreconstructed_MICA_dataset        # Bernardo
from data_loader.loader_reconstructed_HRN import ms1mv3_3Dreconstructed_HRN_dataset        # Bernardo


# os.environ["CUDA_VISIBLE_DEVICES"]='-1' # cpu
# os.environ["CUDA_VISIBLE_DEVICES"]='0'  # gpu
# os.environ["CUDA_VISIBLE_DEVICES"]='1'  # gpu


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg_angmargin', help='Model name [default: pointnet2_cls_ssg]')
parser.add_argument('--log_dir', default='', help='Log dir [default: log]')
# parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')    # original
parser.add_argument('--num_point', type=int, default=2900, help='Point Number [default: 1024]')      # Bernardo
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 16]')  # original
# parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 32]')    # Bernardo
parser.add_argument('--learning_rate', type=float, default=5e-04, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--margin_arc', type=float, default=0.1, help='Margin m for ArcFace')
parser.add_argument('--scale_arc', type=float, default=32, help='Scale s for ArcFace')

# parser.add_argument('--normal', action='store_true', help='Whether to use normal information')     # original
# parser.add_argument('--normal', type=bool, default=True, help='Whether to use normal information')   # Bernardo
parser.add_argument('--normal', type=bool, default=False, help='Whether to use normal information')   # Bernardo

# parser.add_argument('--dataset', type=str, default='frgc', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='synthetic_gpmm', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_lfw', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_ms1mv2', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_ms1mv2_reduced', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_ms1mv2_1000subj', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_ms1mv2_2000subj', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_ms1mv2_5000subj', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_ms1mv2_10000subj', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_webface_1000subj', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_webface_2000subj', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_webface_5000subj', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_webface_10000subj', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_hrn_ms1mv3_reduced', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_hrn_ms1mv3_1000subj', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='[\'mica_casia\',\'mica_ffhq\']', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='mica_casia_100class', help='Name of dataset to train model')   # Bernardo
parser.add_argument('--dataset', type=str, default='mica_casia_5000class', help='Name of dataset to train model')   # Bernardo

FLAGS = parser.parse_args()
if FLAGS.dataset.startswith('[') and FLAGS.dataset.endswith(']'):
    FLAGS.dataset = ast.literal_eval(FLAGS.dataset)
print('FLAGS:', FLAGS)

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, '../models', FLAGS.model+'.py')

if FLAGS.log_dir == '':
    # classes=5000_backbone=resnet_v2_m_50_epoch-num=100_loss=arcface_s=64.0_m=0.5_moment=0.9_batch=64_lr-init=0.1_20230518-214716
    FLAGS.log_dir += 'dataset=' + str(FLAGS.dataset).replace(' ', '').replace('\'', '')
    FLAGS.log_dir += '_model=' + str(FLAGS.model)
    FLAGS.log_dir += '_max_epoch=' + str(FLAGS.max_epoch)
    FLAGS.log_dir += '_lr-init=' + str(FLAGS.learning_rate)
    FLAGS.log_dir += '_moment=' + str(FLAGS.momentum)
    FLAGS.log_dir += '_loss=' + str('arcface')
    FLAGS.log_dir += '_s=' + str(FLAGS.scale_arc)
    FLAGS.log_dir += '_m=' + str(FLAGS.margin_arc)
else:
    FLAGS.log_dir = 'log_face_recognition'
LOG_DIR = os.path.dirname(os.path.abspath(__file__)) + '/logs_training/classification/' + FLAGS.log_dir + '_' + datetime.fromtimestamp(time.time()).strftime("%d%m%Y_%H%M%S")   # Bernardo

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (THIS_FILE_NAME, LOG_DIR)) # bkp of train procedure
LOG_FILE_NAME = 'log_train.txt'
LOG_FOUT = open(os.path.join(LOG_DIR, LOG_FILE_NAME), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

TRAIN_SAMPLES_FILE_NAME = 'samples_train.txt'
TEST_SAMPLES_FILE_NAME  = 'samples_test.txt'

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

BEST_MEAN_LOSS = float('inf')
BEST_ACC = float('-inf')

if type(FLAGS.dataset) == str:
    if FLAGS.dataset.upper() == 'frgc'.upper() or FLAGS.dataset.upper() == 'frgcv2'.upper():
        DATA_PATH = os.path.join(ROOT_DIR, '../data/FRGCv2.0/FRGC-2.0-dist')
        TRAIN_DATASET = frgc2_dataset.FRGCv2_Dataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        TEST_DATASET  = frgc2_dataset.FRGCv2_Dataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

    elif FLAGS.dataset.upper() == 'synthetic_gpmm'.upper():
        DATA_PATH = os.path.join(ROOT_DIR, '../../3DFacePointCloudNet/Data/TrainData')
        n_classes = 100
        n_expressions = 10
        TRAIN_DATASET = synthetic_faces_gpmm_dataset.SyntheticFacesGPMM_Dataset(root=DATA_PATH, npoints=NUM_POINT, num_classes=n_classes, num_expressions=n_expressions, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        TEST_DATASET  = synthetic_faces_gpmm_dataset.SyntheticFacesGPMM_Dataset(root=DATA_PATH, npoints=NUM_POINT, num_classes=n_classes, num_expressions=n_expressions, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

    elif FLAGS.dataset.upper() == 'reconst_mica_lfw'.upper():
        DATA_PATH = os.path.join(ROOT_DIR, '../../MICA/demo/output/lfw')
        min_samples, max_samples = 3, -1
        TRAIN_DATASET = lfw_3Dreconstructed_MICA_dataset.LFR_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        TEST_DATASET  = lfw_3Dreconstructed_MICA_dataset.LFR_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

    elif FLAGS.dataset.upper() == 'reconst_mica_ms1mv2'.upper():
        min_samples, max_samples = 2, -1
        DATA_PATH = os.path.join(ROOT_DIR, '../../MICA/demo/output/MS-Celeb-1M/ms1m-retinaface-t1/images')
        print('Loading train data...')
        TRAIN_DATASET = ms1mv2_3Dreconstructed_MICA_dataset.MS1MV2_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        print('Loading test data...')
        TEST_DATASET  = ms1mv2_3Dreconstructed_MICA_dataset.MS1MV2_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

    elif FLAGS.dataset.upper() == 'reconst_mica_ms1mv2_reduced'.upper():
        min_samples, max_samples = 2, -1
        DATA_PATH = os.path.join(ROOT_DIR, '/home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/../../BOVIFOCR_MICA_3Dreconstruction/demo/output/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images_22subj')
        print('Loading train data...')
        TRAIN_DATASET = ms1mv2_3Dreconstructed_MICA_dataset.MS1MV2_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        print('Loading test data...')
        TEST_DATASET  = ms1mv2_3Dreconstructed_MICA_dataset.MS1MV2_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

    elif FLAGS.dataset.upper() == 'reconst_mica_ms1mv2_1000subj'.upper():
        min_samples, max_samples = 2, -1
        DATA_PATH = os.path.join('/home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/../../BOVIFOCR_MICA_3Dreconstruction/demo/output/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images_1000subj')  # duo
        # DATA_PATH = os.path.join('/home/bjgbiesseck/datasets/MS-Celeb-1M/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images_1000subj')  # diolkos
        # DATA_PATH = os.path.join('/nobackup1/bjgbiesseck/datasets/MS-Celeb-1M_3D_reconstruction_originalMICA/images_1000subj')  # peixoto
        print('Loading train data...')
        TRAIN_DATASET = ms1mv2_3Dreconstructed_MICA_dataset.MS1MV2_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        print('Loading test data...')
        TEST_DATASET  = ms1mv2_3Dreconstructed_MICA_dataset.MS1MV2_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

    elif FLAGS.dataset.upper() == 'reconst_mica_ms1mv2_2000subj'.upper():
        min_samples, max_samples = 2, -1
        # DATA_PATH = os.path.join('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images_2000subj')  # duo
        # DATA_PATH = os.path.join('/home/bjgbiesseck/datasets/MS-Celeb-1M/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images_2000subj')  # diolkos
        DATA_PATH = os.path.join('/nobackup1/bjgbiesseck/datasets/MS-Celeb-1M_3D_reconstruction_originalMICA/images_2000subj')  # peixoto
        print('Loading train data...')
        TRAIN_DATASET = ms1mv2_3Dreconstructed_MICA_dataset.MS1MV2_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        print('Loading test data...')
        TEST_DATASET  = ms1mv2_3Dreconstructed_MICA_dataset.MS1MV2_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

    elif FLAGS.dataset.upper() == 'reconst_mica_ms1mv2_5000subj'.upper():
        min_samples, max_samples = 2, -1
        DATA_PATH = os.path.join('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images_5000subj')  # duo
        # DATA_PATH = os.path.join('/home/bjgbiesseck/datasets/MS-Celeb-1M/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images_5000subj')  # diolkos
        # DATA_PATH = os.path.join('/nobackup1/bjgbiesseck/datasets/MS-Celeb-1M_3D_reconstruction_originalMICA/images_5000subj')  # peixoto
        print('Loading train data...')
        TRAIN_DATASET = ms1mv2_3Dreconstructed_MICA_dataset.MS1MV2_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        print('Loading test data...')
        TEST_DATASET  = ms1mv2_3Dreconstructed_MICA_dataset.MS1MV2_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

    elif FLAGS.dataset.upper() == 'reconst_mica_ms1mv2_10000subj'.upper():
        min_samples, max_samples = 2, -1
        # DATA_PATH = os.path.join('/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images_10000subj')  # duo
        # DATA_PATH = os.path.join('/home/bjgbiesseck/datasets/MS-Celeb-1M/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images_10000subj')  # diolkos
        DATA_PATH = os.path.join('/nobackup1/bjgbiesseck/datasets/MS-Celeb-1M_3D_reconstruction_originalMICA/images_10000subj')  # peixoto
        print('Loading train data...')
        TRAIN_DATASET = ms1mv2_3Dreconstructed_MICA_dataset.MS1MV2_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        print('Loading test data...')
        TEST_DATASET  = ms1mv2_3Dreconstructed_MICA_dataset.MS1MV2_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

    elif FLAGS.dataset.upper() == 'reconst_mica_webface_1000subj'.upper():
        min_samples, max_samples = 2, -1
        # DATA_PATH = os.path.join('/datasets1/bjgbiesseck/WebFace260M/3D_reconstruction/images_1000subj')  # duo
        DATA_PATH = os.path.join('/nobackup/unico/datasets/face_recognition/MICA_3Dreconstruction/WebFace260M_3D_reconstruction_originalMICA/images_1000subj')  # diolkos
        # DATA_PATH = os.path.join('/nobackup1/bjgbiesseck/datasets/WebFace260M_3D_reconstruction_originalMICA/images_1000subj')  # peixoto
        print('Loading train data...')
        TRAIN_DATASET = webface_3Dreconstructed_MICA_dataset.WEBFACE_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        print('Loading test data...')
        TEST_DATASET  = webface_3Dreconstructed_MICA_dataset.WEBFACE_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

    elif FLAGS.dataset.upper() == 'reconst_mica_webface_2000subj'.upper():
        min_samples, max_samples = 2, -1
        # DATA_PATH = os.path.join('/datasets1/bjgbiesseck/WebFace260M/3D_reconstruction/images_2000subj')  # duo
        DATA_PATH = os.path.join('/nobackup/unico/datasets/face_recognition/MICA_3Dreconstruction/WebFace260M_3D_reconstruction_originalMICA/images_2000subj')  # diolkos
        # DATA_PATH = os.path.join('/nobackup1/bjgbiesseck/datasets/WebFace260M_3D_reconstruction_originalMICA/images_2000subj')  # peixoto
        print('Loading train data...')
        TRAIN_DATASET = webface_3Dreconstructed_MICA_dataset.WEBFACE_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        print('Loading test data...')
        TEST_DATASET  = webface_3Dreconstructed_MICA_dataset.WEBFACE_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

    elif FLAGS.dataset.upper() == 'reconst_mica_webface_5000subj'.upper():
        min_samples, max_samples = 2, -1
        # DATA_PATH = os.path.join('/datasets1/bjgbiesseck/WebFace260M/3D_reconstruction/images_5000subj')  # duo
        DATA_PATH = os.path.join('/nobackup/unico/datasets/face_recognition/MICA_3Dreconstruction/WebFace260M_3D_reconstruction_originalMICA/images_5000subj')  # diolkos
        # DATA_PATH = os.path.join('/nobackup1/bjgbiesseck/datasets/WebFace260M_3D_reconstruction_originalMICA/images_5000subj')  # peixoto
        print('Loading train data...')
        TRAIN_DATASET = webface_3Dreconstructed_MICA_dataset.WEBFACE_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        print('Loading test data...')
        TEST_DATASET  = webface_3Dreconstructed_MICA_dataset.WEBFACE_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

    elif FLAGS.dataset.upper() == 'reconst_mica_webface_10000subj'.upper():
        min_samples, max_samples = 2, -1
        # DATA_PATH = os.path.join('/datasets1/bjgbiesseck/WebFace260M/3D_reconstruction/images_10000subj')  # duo
        DATA_PATH = os.path.join('/nobackup/unico/datasets/face_recognition/MICA_3Dreconstruction/WebFace260M_3D_reconstruction_originalMICA/images_10000subj')  # diolkos
        # DATA_PATH = os.path.join('/nobackup1/bjgbiesseck/datasets/WebFace260M_3D_reconstruction_originalMICA/images_10000subj')  # peixoto
        print('Loading train data...')
        TRAIN_DATASET = webface_3Dreconstructed_MICA_dataset.WEBFACE_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        print('Loading test data...')
        TEST_DATASET  = webface_3Dreconstructed_MICA_dataset.WEBFACE_3D_Reconstructed_MICA_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

    elif FLAGS.dataset.upper() == 'reconst_hrn_ms1mv3_reduced'.upper():
        min_samples, max_samples = 2, -1
        DATA_PATH = os.path.join('/datasets1/bjgbiesseck/MS-Celeb-1M/ms1m-retinaface-t1/3D_reconstruction/HRN/images_reduced')  # duo
        print('Loading train data...')
        TRAIN_DATASET = ms1mv3_3Dreconstructed_HRN_dataset.MS1MV3_3D_Reconstructed_HRN_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        print('Loading test data...')
        TEST_DATASET  = ms1mv3_3Dreconstructed_HRN_dataset.MS1MV3_3D_Reconstructed_HRN_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

    elif FLAGS.dataset.upper() == 'reconst_hrn_ms1mv3_1000subj'.upper():
        min_samples, max_samples = 2, -1
        DATA_PATH = os.path.join('/datasets1/bjgbiesseck/MS-Celeb-1M/ms1m-retinaface-t1/3D_reconstruction/HRN/images1000subj')  # duo
        print('Loading train data...')
        TRAIN_DATASET = ms1mv3_3Dreconstructed_HRN_dataset.MS1MV3_3D_Reconstructed_HRN_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        print('Loading test data...')
        TEST_DATASET  = ms1mv3_3Dreconstructed_HRN_dataset.MS1MV3_3D_Reconstructed_HRN_Dataset(root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

    elif FLAGS.dataset.upper() == 'mica_casia'.upper():
        min_samples, max_samples = -1, -1
        TRAIN_DATASET, TEST_DATASET = None, None
        # DATA_PATH = '/datasets2/frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/1_CASIA-WebFace/imgs_crops_112x112/output'  # duo
        DATA_PATH = '/groups/unico/frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/1_CASIA-WebFace/imgs_crops_112x112/output' # daugman
        print(f'\nLoading train data: \'{FLAGS.dataset}\' ...')
        TRAIN_DATASET = casia_3Dreconstructed_MICA_dataset.CASIA_3D_Reconstructed_MICA_Dataset(TRAIN_DATASET, root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        print(f'Loading test data: \'{FLAGS.dataset}\' ...')
        TEST_DATASET  = casia_3Dreconstructed_MICA_dataset.CASIA_3D_Reconstructed_MICA_Dataset(TEST_DATASET, root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

    elif FLAGS.dataset.upper() == 'mica_casia_100class'.upper():
        min_samples, max_samples = -1, -1
        TRAIN_DATASET, TEST_DATASET = None, None
        DATA_PATH = '/datasets2/frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/1_CASIA-WebFace/output_100class'  # duo
        print(f'\nLoading train data: \'{FLAGS.dataset}\' ...')
        TRAIN_DATASET = casia_3Dreconstructed_MICA_dataset.CASIA_3D_Reconstructed_MICA_Dataset(TRAIN_DATASET, root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        print(f'Loading test data: \'{FLAGS.dataset}\' ...')
        TEST_DATASET  = casia_3Dreconstructed_MICA_dataset.CASIA_3D_Reconstructed_MICA_Dataset(TEST_DATASET, root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

    elif FLAGS.dataset.upper() == 'mica_casia_5000class'.upper():
        min_samples, max_samples = -1, -1
        TRAIN_DATASET, TEST_DATASET = None, None
        DATA_PATH = '/datasets2/frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/1_CASIA-WebFace/output_5000class'  # duo
        print(f'\nLoading train data: \'{FLAGS.dataset}\' ...')
        TRAIN_DATASET = casia_3Dreconstructed_MICA_dataset.CASIA_3D_Reconstructed_MICA_Dataset(TRAIN_DATASET, root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        print(f'Loading test data: \'{FLAGS.dataset}\' ...')
        TEST_DATASET  = casia_3Dreconstructed_MICA_dataset.CASIA_3D_Reconstructed_MICA_Dataset(TEST_DATASET, root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

    elif FLAGS.dataset.upper() == 'mica_ffhq'.upper():
        min_samples, max_samples = -1, -1
        TRAIN_DATASET, TEST_DATASET = None, None
        # DATA_PATH = '/datasets2/frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/2_FFHQ/images_crops_112x112/output'  # duo
        DATA_PATH = '/groups/unico/frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/2_FFHQ/images_crops_112x112/output' # daugman
        print(f'\nLoading train data: \'{FLAGS.dataset}\' ...')
        TRAIN_DATASET = ffhq_3Dreconstructed_MICA_dataset.FFHQ_3D_Reconstructed_MICA_Dataset(TRAIN_DATASET, root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
        print(f'Loading test data: \'{FLAGS.dataset}\' ...')
        TEST_DATASET  = ffhq_3Dreconstructed_MICA_dataset.FFHQ_3D_Reconstructed_MICA_Dataset(TEST_DATASET, root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)


elif type(FLAGS.dataset) == list:
    TRAIN_DATASET, TEST_DATASET = None, None
    for i, dataset in enumerate(FLAGS.dataset):
        if dataset.upper() == 'mica_casia'.upper():
            min_samples, max_samples = -1, -1
            # DATA_PATH = '/datasets2/frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/1_CASIA-WebFace/imgs_crops_112x112/output'  # duo
            DATA_PATH = '/groups/unico/frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/1_CASIA-WebFace/imgs_crops_112x112/output' # daugman
            print(f'\nLoading train data: \'{dataset}\' ...')
            TRAIN_DATASET = casia_3Dreconstructed_MICA_dataset.CASIA_3D_Reconstructed_MICA_Dataset(TRAIN_DATASET, root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
            print(f'Loading test data: \'{dataset}\' ...')
            TEST_DATASET  = casia_3Dreconstructed_MICA_dataset.CASIA_3D_Reconstructed_MICA_Dataset(TEST_DATASET, root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

        elif dataset.upper() == 'mica_ffhq'.upper():
            min_samples, max_samples = -1, -1
            # DATA_PATH = '/datasets2/frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/2_FFHQ/images_crops_112x112/output'  # duo
            DATA_PATH = '/groups/unico/frcsyn_wacv2024/datasets/3D_reconstruction_MICA/real/2_FFHQ/images_crops_112x112/output' # daugman
            print(f'\nLoading train data: \'{dataset}\' ...')
            TRAIN_DATASET = ffhq_3Dreconstructed_MICA_dataset.FFHQ_3D_Reconstructed_MICA_Dataset(TRAIN_DATASET, root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
            print(f'Loading test data: \'{dataset}\' ...')
            TEST_DATASET  = ffhq_3Dreconstructed_MICA_dataset.FFHQ_3D_Reconstructed_MICA_Dataset(TEST_DATASET, root=DATA_PATH, npoints=NUM_POINT, min_samples=min_samples, max_samples=max_samples, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)


# Bernardo
assert TRAIN_DATASET.num_classes == TEST_DATASET.num_classes
NUM_CLASSES = TRAIN_DATASET.num_classes

def save_train_test_samples(samples_list, path_output_file):
    with open(path_output_file, 'w') as file_handler:
        for item in samples_list:
            file_handler.write("{}\n".format(item))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            print('\nBuilding input tensors...')
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            print('Building backbone...')
            # pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)                         # original
            interm_layers, embd, logits, end_points, weights_fc3 = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay, num_class=NUM_CLASSES)    # Bernardo
            print('Building classification layer...')
            logits, loss, classify_loss = MODEL.get_loss_arcface(logits, labels_pl, end_points, weights_fc3, TRAIN_DATASET.num_classes, FLAGS.margin_arc, float(FLAGS.scale_arc))
            # pred = embds   # TESTE
            # pred, loss, classify_loss = MODEL.get_loss_common_cross_entropy(embd, labels_pl, end_points, weights_fc3, TRAIN_DATASET.num_classes)

            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
            for l in losses + [total_loss]:
                tf.summary.scalar(l.op.name, l)

            correct = tf.equal(tf.argmax(logits, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # print "--- Get training operator"    # original
            print("--- Get training operator")     # Bernardo (for python 3.7)
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        print('Initializing global variables...')
        init = tf.global_variables_initializer()
        print('sess.run(init) --> Starting session...')
        sess.run(init)
        print('    session started!')

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'interm_layers': interm_layers,
               'embd': embd,
               'logits': logits,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        global EPOCH_CNT, BEST_MEAN_LOSS, BEST_ACC
        best_acc = -1
        for epoch in range(MAX_EPOCH+1):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            # Commented to reduce time when training with 10,000 classes
            # if epoch > 0:
            #     train_one_epoch(sess, ops, train_writer)

            # Modified to reduce time when training with 10,000 classes
            print('Starting train_one_epoch...')
            loss_sum, train_mean_loss, train_accuracy, train_avg_class_acc = train_one_epoch(epoch, sess, ops, train_writer)

            # train_one_epoch(sess, ops, train_writer)
            # eval_one_epoch(sess, ops, test_writer)
            # loss_sum, train_mean_loss, train_accuracy, train_avg_class_acc = eval_train_one_epoch(sess, ops, train_writer)   # Commented to reduce time when training with 10,000 classes
            loss_sum, test_mean_loss, test_accuracy, test_avg_class_acc = eval_test_one_epoch(epoch, sess, ops, test_writer)
            log_string('')

            if train_mean_loss < BEST_MEAN_LOSS:
                BEST_MEAN_LOSS = train_mean_loss
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model_best_train_mean_loss.ckpt"))
                print("Best model (train_mean_loss) saved in file: %s" % save_path)
            if train_accuracy > BEST_ACC:
                BEST_ACC = train_accuracy
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model_best_train_accuracy.ckpt"))
                print("Best model (train_accuracy) saved in file: %s" % save_path)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

            # Bernardo
            plot_classification_training_history()
            EPOCH_CNT += 1


def save_samples(path_save, batch_data, out_layer1_l1_xyz, out_layer2_l2_xyz, out_layer3_l3_xyz):
    os.makedirs(path_save, exist_ok=True)
    # print(f'batch_data.shape: {batch_data.shape}')
    # print(f'out_layer1_l1_xyz.shape: {out_layer1_l1_xyz.shape}')
    # print(f'out_layer2_l2_xyz.shape: {out_layer2_l2_xyz.shape}')
    # print(f'out_layer3_l3_xyz.shape: {out_layer3_l3_xyz.shape}')

    for i in range(batch_data.shape[0]):
        print(f'Saving sample {i}/{batch_data.shape[0]-1}', end='\r')

        path_input_pc = os.path.join(path_save, f'sample{i}_input_pc.obj')
        write_obj(path_input_pc, batch_data[i])

        path_out_layer1_l1_xyz = os.path.join(path_save, f'sample{i}_out_layer1_l1_xyz.obj')
        write_obj(path_out_layer1_l1_xyz, out_layer1_l1_xyz[i])

        path_out_layer2_l2_xyz = os.path.join(path_save, f'sample{i}_out_layer2_l2_xyz.obj')
        write_obj(path_out_layer2_l2_xyz, out_layer2_l2_xyz[i])

        path_out_layer3_l3_xyz = os.path.join(path_save, f'sample{i}_out_layer3_l3_xyz.obj')
        write_obj(path_out_layer3_l3_xyz, out_layer3_l3_xyz[i])

    print('')
    # sys.exit(0)


def train_one_epoch(epoch, sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TRAIN_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    num_batches = TRAIN_DATASET.__len__() / BATCH_SIZE

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d TRAINING ----'%(EPOCH_CNT))

    while TRAIN_DATASET.has_next_batch():
        print('batch_idx: %d/%d' % (batch_idx, num_batches), end='\r')

        # batch_data, batch_label = TRAIN_DATASET.next_batch(augment=True)   # original
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=False)    # Bernardo

        #batch_data = provider.random_point_dropout(batch_data)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, embd, logits, \
            out_layer1_l1_xyz, out_layer2_l2_xyz, out_layer3_l3_xyz = sess.run([ops['merged'], ops['step'],
                                                                                ops['train_op'], ops['loss'], ops['embd'], ops['logits'],
                                                                                ops['interm_layers']['layer1']['l1_xyz'],
                                                                                ops['interm_layers']['layer2']['l2_xyz'],
                                                                                ops['interm_layers']['layer3']['l3_xyz']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        logits = np.argmax(logits, 1)
        correct = np.sum(logits[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        
        for i in range(0, bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (logits[i] == l)

        if epoch % 10 == 0 and batch_idx == 0:
            dir_samples = f'train_samples/epoch{epoch}/batch{batch_idx}'
            path_save = os.path.join(LOG_DIR, dir_samples)
            save_samples(path_save, batch_data, out_layer1_l1_xyz, out_layer2_l2_xyz, out_layer3_l3_xyz)

        batch_idx += 1

    print('')

    train_mean_loss = loss_sum / float(batch_idx)
    train_accuracy = total_correct / float(total_seen)
    train_avg_class_acc = np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))
    log_string('train loss sum: %f' % (loss_sum))
    log_string('train mean loss: %f' % (train_mean_loss))
    log_string('train accuracy: %f'% (train_accuracy))
    log_string('train avg class acc: %f' % (train_avg_class_acc))

    TRAIN_DATASET.reset()
    return loss_sum, train_mean_loss, train_accuracy, train_avg_class_acc



def eval_train_one_epoch(epoch, sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TRAIN_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    num_batches = TRAIN_DATASET.__len__() / BATCH_SIZE

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d TRAIN EVALUATION ----'%(EPOCH_CNT))

    while TRAIN_DATASET.has_next_batch():
        print('batch_idx: %d/%d' % (batch_idx, num_batches), end='\r')

        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['logits']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1
        for i in range(0, bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)

    print('')
    
    train_mean_loss = loss_sum / float(batch_idx)
    train_accuracy = total_correct / float(total_seen)
    train_avg_class_acc = np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))
    log_string('train loss sum: %f' % (loss_sum))
    log_string('train mean loss: %f' % (train_mean_loss))
    log_string('train accuracy: %f'% (train_accuracy))
    log_string('train avg class acc: %f' % (train_avg_class_acc))
    
    # EPOCH_CNT += 1
    TRAIN_DATASET.reset()
    return loss_sum, train_mean_loss, train_accuracy, train_avg_class_acc



def eval_test_one_epoch(epoch, sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    num_batches = TEST_DATASET.__len__() / BATCH_SIZE
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d TEST EVALUATION ----'%(EPOCH_CNT))
    
    while TEST_DATASET.has_next_batch():
        print('batch_idx: %d/%d' % (batch_idx, num_batches), end='\r')

        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]

        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['is_training_pl']: is_training}

        # summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
        #     ops['loss'], ops['logits']], feed_dict=feed_dict)
        summary, step, loss_val, pred_val, \
            out_layer1_l1_xyz, out_layer2_l2_xyz, out_layer3_l3_xyz = sess.run([ops['merged'], ops['step'],
                                                                                ops['loss'], ops['logits'],
                                                                                ops['interm_layers']['layer1']['l1_xyz'],
                                                                                ops['interm_layers']['layer2']['l2_xyz'],
                                                                                ops['interm_layers']['layer3']['l3_xyz']], feed_dict=feed_dict)
        
            
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        
        for i in range(0, bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)

        if epoch % 10 == 0 and batch_idx == 0:
            dir_samples = f'test_samples/epoch{epoch}/batch{batch_idx}'
            path_save = os.path.join(LOG_DIR, dir_samples)
            save_samples(path_save, batch_data, out_layer1_l1_xyz, out_layer2_l2_xyz, out_layer3_l3_xyz)

        batch_idx += 1
    
    test_mean_loss = loss_sum / float(batch_idx)
    test_accuracy = total_correct / float(total_seen)
    test_avg_class_acc = np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))
    log_string('test loss sum: %f' % (loss_sum))
    log_string('test mean loss: %f' % (test_mean_loss))
    log_string('test accuracy: %f'% (test_accuracy))
    log_string('test avg class acc: %f' % (test_avg_class_acc))
    
    # EPOCH_CNT += 1
    TEST_DATASET.reset()
    return loss_sum, test_mean_loss, test_accuracy, test_avg_class_acc


# Bernardo
def plot_classification_training_history():
    path_log_file = os.path.join(LOG_DIR, LOG_FILE_NAME)
    parameters, epoch, train_mean_loss, train_accuracy, test_mean_loss, test_accuracy = plots_fr_pointnet2.load_original_training_log_pointnet2_angmargin(path_file=path_log_file)
    
    if FLAGS.dataset.upper() == 'frgc'.upper() or FLAGS.dataset.upper() == 'frgcv2'.upper():
        title = 'PointNet++ training on FRGCv2 \nClassification (1:N) - '+str(NUM_CLASSES)+' classes'
    elif FLAGS.dataset.upper() == 'synthetic_gpmm'.upper():
        title = 'PointNet++ training on SyntheticFaces \nClassification (1:N) - '+str(NUM_CLASSES)+' classes - '+str(n_expressions)+' expressions'
    elif FLAGS.dataset.upper() == 'reconst_mica_lfw'.upper():
        title = 'PointNet++ training on LFW-Reconst3D-MICA \nClassification (1:N) - '+str(NUM_CLASSES)+' classes - min_samples='+str(min_samples)+' - max_samples='+str(max_samples)
    elif FLAGS.dataset.upper().startswith('reconst_mica_ms1mv2'.upper()):
        title = 'PointNet++ training on ' + FLAGS.dataset.upper() + ' \nClassification (1:N) - '+str(NUM_CLASSES)+' classes - min_samples='+str(min_samples)+' - max_samples='+str(max_samples)
    elif FLAGS.dataset.upper().startswith('reconst_mica_webface'.upper()):
        title = 'PointNet++ training on ' + FLAGS.dataset.upper() + ' \nClassification (1:N) - '+str(NUM_CLASSES)+' classes - min_samples='+str(min_samples)+' - max_samples='+str(max_samples)
    else:
        title = 'PointNet++ training'


    subtitle = 'Parameters: ' + plots_fr_pointnet2.break_string(parameters, substring=', ', num_parts=4)
    # path_image = './training_history.png'
    path_image = '/'.join(path_log_file.split('/')[:-1]) + '/training_history_from_log_file.png'
    print('Saving training history:', path_image)
    plots_fr_pointnet2.plot_training_history_pointnet2_angmargin(epoch, train_mean_loss, train_accuracy, test_mean_loss, test_accuracy, title=title, subtitle=subtitle, path_image=path_image, show_fig=False, save_fig=True)



if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))

    # Bernardo
    save_train_test_samples(TRAIN_DATASET.datapath, os.path.join(LOG_DIR, TRAIN_SAMPLES_FILE_NAME))
    save_train_test_samples(TEST_DATASET.datapath, os.path.join(LOG_DIR, TEST_SAMPLES_FILE_NAME))

    train()
    LOG_FOUT.close()

    # Bernardo
    plot_classification_training_history()

    print('\nFinished!\n')
