'''
    Evaluate classification performance with optional voting.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import pickle
import scipy.misc
from scipy import interpolate
from sklearn.model_selection import KFold
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, '../models'))
sys.path.append(os.path.join(ROOT_DIR, '../utils'))
import provider

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import timeit
import sklearn
import argparse
import cv2
import numpy as np
import torch
from skimage import transform as trans
# from backbones import get_model
from sklearn.metrics import roc_curve, auc

# from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from prettytable import PrettyTable
from pathlib import Path
import warnings
import yaml

def sample_colours_from_colourmap(n_colours, colour_map):
    import matplotlib.pyplot as plt

    cm = plt.get_cmap(colour_map)
    return [cm(1.0 * i / n_colours)[:3] for i in range(n_colours)]

'''
from data_loader.loader_reconstructed_MICA import lfw_evaluation_3Dreconstructed_MICA_dataset_pairs      # Bernardo
from data_loader.loader_reconstructed_MICA import magVerif_pairs_3Dreconstructed_MICA                    # Bernardo
from data_loader.loader_reconstructed_MICA import calfw_evaluation_3Dreconstructed_MICA_dataset_pairs    # Bernardo
'''

parser = argparse.ArgumentParser(description='do ijb test')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
# parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name [default: pointnet2_cls_ssg]')  # original
parser.add_argument('--model', default='pointnet2_cls_ssg_angmargin', help='Model name [default: pointnet2_cls_ssg_angmargin]')    # Bernardo
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 16]')
# parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')    # original
parser.add_argument('--num_point', type=int, default=2900, help='Point Number [default: 1024]')      # Bernardo

# parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')   # original
# parser.add_argument('--model_path', default='logs_training/classification/log_face_recognition_train_arcface=ms1mv2-1000subj_batch=16_margin=0.0/model_best_train_accuracy.ckpt', help='model checkpoint file path')  # Bernardo
# parser.add_argument('--model_path', default='logs_training/classification/log_face_recognition_train_arcface=ms1mv2-2000subj_batch=16_margin=0.0/model_best_train_accuracy.ckpt', help='model checkpoint file path')  # Bernardo
# parser.add_argument('--model_path', default='logs_training/classification/log_face_recognition_train_arcface=ms1mv2-5000subj_batch=16_margin=0.0/model_best_train_accuracy.ckpt', help='model checkpoint file path')  # Bernardo
# parser.add_argument('--model_path', default='logs_training/classification/log_face_recognition_train_arcface=ms1mv2-1000subj_batch=16_margin=0.0_classification-layer=1/model_best_train_accuracy.ckpt', help='model checkpoint file path')  # Bernardo
parser.add_argument('--model_path', default='logs_training/classification/dataset=reconst_mica_ms1mv2_1000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=32_m=0.0_06052023_114705/model_best_train_accuracy.ckpt', help='model checkpoint file path')  # Bernardo
# parser.add_argument('--model_path', default='logs_training/classification/dataset=reconst_mica_ms1mv2_2000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=32_m=0.0_09062023_184940/model_best_train_accuracy.ckpt', help='model checkpoint file path')  # Bernardo
# parser.add_argument('--model_path', default='logs_training/classification/dataset=reconst_mica_ms1mv2_5000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-06_moment=0.9_loss=arcface_s=32_m=0.0_12062023_154451/model_best_train_accuracy.ckpt', help='model checkpoint file path')  # Bernardo
# parser.add_argument('--model_path', default='logs_training/classification/dataset=reconst_mica_ms1mv2_10000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=32_m=0.0_26052023_222414/model_best_train_accuracy.ckpt', help='model checkpoint file path')  # Bernardo
# parser.add_argument('--model_path', default='logs_training/classification/dataset=reconst_mica_webface_1000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=16_m=0.0_05062023_194932/model_best_train_accuracy.ckpt', help='model checkpoint file path')  # Bernardo
# parser.add_argument('--model_path', default='logs_training/classification/dataset=reconst_mica_webface_2000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=16_m=0.0_05062023_213735/model_best_train_accuracy.ckpt', help='model checkpoint file path')  # Bernardo
# parser.add_argument('--model_path', default='logs_training/classification/dataset=reconst_mica_webface_5000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=32_m=0.0_06062023_235151/model_best_train_accuracy.ckpt', help='model checkpoint file path')  # Bernardo
# parser.add_argument('--model_path', default='logs_training/classification/dataset=reconst_mica_webface_10000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-06_moment=0.9_loss=arcface_s=32_m=0.0_13062023_123431/model_best_train_accuracy.ckpt', help='model checkpoint file path')  # Bernardo

parser.add_argument('--num_class', type=int, default=1000, help='Number of training and testing classes')      # Bernardo

parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
# parser.add_argument('--normal', action='store_true', help='Whether to use normal information')      # original
parser.add_argument('--normal', type=bool, default=False, help='Whether to use normal information')   # Bernardo
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores from multiple rotations [default: 1]')
parser.add_argument('--margin', type=float, default=0.5, help='Minimum distance for non-corresponding pairs in Contrastive Loss')

# parser.add_argument('--dataset', type=str, default='frgc', help='Name of dataset to train model')                 # Bernardo
# parser.add_argument('--dataset', type=str, default='synthetic_gpmm', help='Name of dataset to train model')       # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_ms1mv2', help='Name of dataset to train model')  # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_lfw', help='Name of dataset to train model')     # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_agedb', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_cfp', help='Name of dataset to train model')     # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_calfw', help='Name of dataset to train model')   # Bernardo

FLAGS = parser.parse_args()

NUM_CLASSES = FLAGS.num_class
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MARGIN = FLAGS.margin

MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

HOSTNAME = socket.gethostname()



target = 'IJBC'
# config_path = args.config_path
model_path = FLAGS.model_path
# image_path = '/datasets1/bjgbiesseck/IJB-C/rec_data_ijbc/'
image_path = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/IJB-C/IJB/IJB-C/rec_data_ijbc'
protocols_path = '/datasets1/bjgbiesseck/IJB-C/rec_data_ijbc'
result_dir = 'results_ijbc'
gpu_id = None
use_norm_score = True  # if Ture, TestMode(N1)
# use_detector_score = True  # if Ture, TestMode(D1)
use_detector_score = False   # if Ture, TestMode(D1)
use_flip_test = True  # if Ture, TestMode(F1)
job = 'insightface'
# batch_size = 128


class Embedding(object):
    def __init__(self, prefix, data_shape, batch_size=1):
        image_size = (112, 112)
        self.image_size = image_size

        # original
        # weight = torch.load(prefix)
        # resnet = get_model(args.network, dropout=0, fp16=False).cuda()
        # resnet.load_state_dict(weight)
        # model = torch.nn.DataParallel(resnet)
        # self.model = model
        # self.model.eval()

        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        src[:, 0] += 8.0
        self.src = src
        self.batch_size = batch_size
        self.data_shape = data_shape

    def get(self, rimg, landmark):
        assert landmark.shape[0] == 68 or landmark.shape[0] == 5
        assert landmark.shape[1] == 2
        if landmark.shape[0] == 68:
            landmark5 = np.zeros((5, 2), dtype=np.float32)
            landmark5[0] = (landmark[36] + landmark[39]) / 2
            landmark5[1] = (landmark[42] + landmark[45]) / 2
            landmark5[2] = landmark[30]
            landmark5[3] = landmark[48]
            landmark5[4] = landmark[54]
        else:
            landmark5 = landmark
        tform = trans.SimilarityTransform()
        tform.estimate(landmark5, self.src)
        M = tform.params[0:2, :]
        img = cv2.warpAffine(rimg,
                             M, (self.image_size[1], self.image_size[0]),
                             borderValue=0.0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img)
        # img = np.transpose(img, (2, 0, 1))  # 3*112*112, RGB
        # img_flip = np.transpose(img_flip, (2, 0, 1))
        # input_blob = np.zeros((2, 3, self.image_size[1], self.image_size[0]), dtype=np.uint8)
        input_blob = np.zeros((2, self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        input_blob[0] = img
        input_blob[1] = img_flip
        return input_blob

    @torch.no_grad()
    def forward_db(self, batch_data, sess, ops):
        is_training = False
        # imgs = torch.Tensor(batch_data).cuda()

        bsize = batch_data.shape[0]
        # print('bsize:', bsize)

        cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 3), dtype=np.float32)
        cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

        cur_batch_data[0:bsize,:] = batch_data

        feed_dict0 = {ops['pointclouds_pl']: cur_batch_data,
                      ops['labels_pl']: cur_batch_label,
                      ops['is_training_pl']: is_training}
        
        cur_embd, pred = sess.run([ops['embds'], ops['pred']], feed_dict=feed_dict0)
        # feat = cur_embd

        # imgs = torch.Tensor(batch_data)
        # imgs.div_(255).sub_(0.5).div_(0.5)
        
        # # feat = self.model(imgs)
        # cur_embd = sess.run(embds_ph, feed_dict={images: batch_data, train_ph_dropout: False, train_ph_bn: False})
        feat = cur_embd[0:bsize,:]

        # print('cur_batch_data:', cur_batch_data)
        # print('batch_data.shape:', batch_data.shape)
        # print('feat:', feat)
        # print('feat.shape:', feat.shape)

        if np.sum(np.isnan(batch_data)) > 0:
            print('Error, nan found')
            print('batch_data:', batch_data)
            print('batch_data.shape:', batch_data.shape)
            sys.exit(0)
        
        if np.sum(np.isnan(cur_embd)) > 0:
            print('Error, nan found')
            print('cur_embd:', cur_embd)
            print('cur_embd.shape:', cur_embd.shape)
            sys.exit(0)

        # feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])   # original
        # feat = feat.reshape([self.batch_size, feat.shape[1]])       # Bernardo
        # return feat.cpu().numpy()
        return feat



def divideIntoNstrand(listTemp, n):
    twoList = [[] for i in range(n)]
    for i, e in enumerate(listTemp):
        twoList[i % n].append(e)
    return twoList


def read_template_media_list(path):
    # ijb_meta = np.loadtxt(path, dtype=str)
    ijb_meta = pd.read_csv(path, sep=' ', header=None).values
    templates = ijb_meta[:, 1].astype(np.int)
    medias = ijb_meta[:, 2].astype(np.int)
    return templates, medias


# In[ ]:


def read_template_pair_list(path):
    # pairs = np.loadtxt(path, dtype=str)
    pairs = pd.read_csv(path, sep=' ', header=None).values
    # print(pairs.shape)
    # print(pairs[:, 0].astype(np.int))
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    label = pairs[:, 2].astype(np.int)
    return t1, t2, label


# In[ ]:


def read_image_feature(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


def pc_normalize(pc):
    # Bernardo
    pc /= 100
    pc = (pc - pc.min()) / (pc.max() - pc.min())

    # l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m

    return pc


# In[ ]:
def get_image_feature(img_path, files_list, model_path, epoch, gpu_id):
    batch_size = FLAGS.batch_size
    # data_shape = (3, 112, 112)
    # data_shape = (112, 112, 3)
    data_shape = (2900, 3)

    files = files_list
    print('files:', len(files))
    rare_size = len(files) % batch_size
    faceness_scores = []
    batch = 0
    # img_feats = np.empty((len(files), 1024), dtype=np.float32)
    img_feats = np.empty((len(files), NUM_CLASSES), dtype=np.float32)

    '''
    # Bernardo
    config = yaml.load(open(args.config_path))
    images = tf.placeholder(dtype=tf.float32, shape=[None, config['image_size'], config['image_size'], 3], name='input_image')
    train_phase_dropout = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase')
    train_phase_bn = tf.placeholder(dtype=tf.bool, shape=None, name='train_phase_last')
    embds, _ = get_embd(images, train_phase_dropout, train_phase_bn, config)
    print('done!')
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        tf.global_variables_initializer().run()
        print('loading model:', model_path)
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        print('done!')
        # Bernardo
    '''

    is_training = False
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        # pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        embd, end_points, weights_fc3 = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=None, num_class=NUM_CLASSES)    # Bernardo
        
        # MODEL.get_loss(pred, labels_pl, end_points)
        embds, pred, loss, classify_loss = MODEL.get_loss_arcface(embd, labels_pl, end_points, weights_fc3, num_classes=NUM_CLASSES)
        
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Restore variables from disk.
        print("\nLoading model:", MODEL_PATH)
        saver.restore(sess, MODEL_PATH)
        print('Model restored.')

        ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'embds': embds,
           'pred': pred,
           'loss': total_loss,
           'end_points': end_points}



        # batch_data = np.empty((2 * batch_size, 3, 112, 112))  # original
        # batch_data = np.empty((2 * batch_size, 112, 112, 3))  # Bernardo
        batch_data = np.empty((batch_size, 2900, 3))            # Bernardo
        embedding = Embedding(model_path, data_shape, batch_size)
        for img_index, each_line in enumerate(files[:len(files) - rare_size]):
            name_lmk_score = each_line.strip().split(' ')
            
            # img_name = os.path.join(img_path, name_lmk_score[0])                    # original
            file_ext = 'mesh_centralized-nosetip_with-normals_filter-radius=100.npy'  # Bernardo
            folder_name = name_lmk_score[0].split('.')[0]                             # Bernardo
            img_name = os.path.join(img_path, folder_name, file_ext)                  # Bernardo

            # print('get_image_feature - img_name:', img_name)

            '''  # original
            img = cv2.imread(img_name)
            lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                        dtype=np.float32)
            lmk = lmk.reshape((5, 2))
            input_blob = embedding.get(img, lmk)
            '''

            # Bernardo
            if img_name.endswith('.npy'):
                point_set = np.load(img_name).astype(np.float32)
            
            if point_set.shape[1] > 3:        # if contains normals and curvature
                point_set = point_set[:,0:3]  # remove normals and curvature
            point_set = pc_normalize(point_set)

            # batch_data[2 * (img_index - batch * batch_size)][:] = input_blob[0]
            # batch_data[2 * (img_index - batch * batch_size) + 1][:] = input_blob[1]
            batch_data[(img_index - batch * batch_size)][:] = point_set[:2900]
            if (img_index + 1) % batch_size == 0:
                print('batch', batch)
                img_feats[batch * batch_size:batch * batch_size +
                                            batch_size][:] = embedding.forward_db(batch_data, sess, ops)
                batch += 1
            faceness_scores.append(name_lmk_score[-1])

        # batch_data = np.empty((2 * rare_size, 3, 112, 112))   # original
        # batch_data = np.empty((2 * rare_size, 112, 112, 3))   # Bernardo
        batch_data = np.empty((rare_size, 2900, 3))            # Bernardo
        embedding = Embedding(model_path, data_shape, rare_size)
        for img_index, each_line in enumerate(files[len(files) - rare_size:]):
            name_lmk_score = each_line.strip().split(' ')

            # img_name = os.path.join(img_path, name_lmk_score[0])    # original
            file_ext = 'mesh_centralized-nosetip_with-normals_filter-radius=100.npy'  # Bernardo
            folder_name = name_lmk_score[0].split('.')[0]                             # Bernardo
            img_name = os.path.join(img_path, folder_name, file_ext)                  # Bernardo

            '''  # original
            img = cv2.imread(img_name)
            lmk = np.array([float(x) for x in name_lmk_score[1:-1]],
                        dtype=np.float32)
            lmk = lmk.reshape((5, 2))
            input_blob = embedding.get(img, lmk)
            '''

            # Bernardo
            if img_name.endswith('.npy'):
                point_set = np.load(img_name).astype(np.float32)

            if point_set.shape[1] > 3:        # if contains normals and curvature
                point_set = point_set[:,0:3]  # remove normals and curvature
            point_set = pc_normalize(point_set)

            # batch_data[2 * img_index][:] = input_blob[0]
            # batch_data[2 * img_index + 1][:] = input_blob[1]
            batch_data[img_index][:] = point_set[:2900]   # Bernardo
            if (img_index + 1) % rare_size == 0:
                print('batch', batch)
                print('len(files) - rare_size:', len(files) - rare_size)
                print('img_feats[len(files)-rare_size:].shape:', img_feats[len(files)-rare_size:].shape)
                img_feats[len(files) -
                        rare_size:][:] = embedding.forward_db(batch_data, sess, ops)
                batch += 1
            faceness_scores.append(name_lmk_score[-1])
        faceness_scores = np.array(faceness_scores).astype(np.float32)
        # img_feats = np.ones( (len(files), 1024), dtype=np.float32) * 0.01
        # faceness_scores = np.ones( (len(files), ), dtype=np.float32 )
        return img_feats, faceness_scores





# In[ ]:


def image2template_feature(img_feats=None, templates=None, medias=None):
    # ==========================================================
    # 1. face image feature l2 normalization. img_feats:[number_image x feats_dim]
    # 2. compute media feature.
    # 3. compute template feature.
    # ==========================================================
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))

    for count_template, uqt in enumerate(unique_templates):

        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias,
                                                       return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [
                    np.mean(face_norm_feats[ind_m], axis=0, keepdims=True)
                ]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, axis=0)
        if count_template % 2000 == 0:
            print('Finish Calculating {} template features.'.format(
                count_template))
    # template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    # print(template_norm_feats.shape)
    return template_norm_feats, unique_templates


# In[ ]:


def verification(template_norm_feats=None,
                 unique_templates=None,
                 p1=None,
                 p2=None):
    # ==========================================================
    #         Compute set-to-set Similarity Score.
    # ==========================================================
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template

    score = np.zeros((len(p1),))  # save cosine distance between pairs

    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


# In[ ]:
def verification2(template_norm_feats=None,
                  unique_templates=None,
                  p1=None,
                  p2=None):
    template2id = np.zeros((max(unique_templates) + 1, 1), dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template
    score = np.zeros((len(p1),))  # save cosine distance between pairs
    total_pairs = np.array(range(len(p1)))
    batchsize = 100000  # small batchsize instead of all pairs in one batch due to the memory limiation
    sublists = [
        total_pairs[i:i + batchsize] for i in range(0, len(p1), batchsize)
    ]
    total_sublists = len(sublists)
    for c, s in enumerate(sublists):
        feat1 = template_norm_feats[template2id[p1[s]]]
        feat2 = template_norm_feats[template2id[p2[s]]]
        similarity_score = np.sum(feat1 * feat2, -1)
        score[s] = similarity_score.flatten()
        if c % 10 == 0:
            print('Finish {}/{} pairs.'.format(c, total_sublists))
    return score


def read_score(path):
    with open(path, 'rb') as fid:
        img_feats = pickle.load(fid)
    return img_feats


exper_id = model_path.split('/')[-2]            # Bernardo
save_path = os.path.join(result_dir, exper_id)  # Bernardo
score_save_file = os.path.join(save_path, "%s.npy" % target.lower())
label_save_file = os.path.join(save_path, "label.npy")

# Bernardo
if not os.path.exists(score_save_file):

    # # Step1: Load Meta Data

    # In[ ]:

    assert target == 'IJBC' or target == 'IJBB'

    # =============================================================
    # load image and template relationships for template feature embedding
    # tid --> template id,  mid --> media id
    # format:
    #           image_name tid mid
    # =============================================================
    start = timeit.default_timer()
    templates, medias = read_template_media_list(
        # os.path.join('%s/meta' % image_path, '%s_face_tid_mid.txt' % target.lower()))      # original
        os.path.join('%s/meta' % protocols_path, '%s_face_tid_mid.txt' % target.lower()))    # Bernardo
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))

    # In[ ]:

    # =============================================================
    # load template pairs for template-to-template verification
    # tid : template id,  label : 1/0
    # format:
    #           tid_1 tid_2 label
    # =============================================================
    start = timeit.default_timer()
    p1, p2, label = read_template_pair_list(
        # os.path.join('%s/meta' % image_path, '%s_template_pair_label.txt' % target.lower()))     # original
        os.path.join('%s/meta' % protocols_path, '%s_template_pair_label.txt' % target.lower()))   # Bernardo
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))



    # # Step 2: Get Image Features

    # In[ ]:

    # =============================================================
    # load image features
    # format:
    #           img_feats: [image_num x feats_dim] (227630, 512)
    # =============================================================
    start = timeit.default_timer()

    # img_path = '%s/loose_crop' % image_path    # original
    # img_path = '%s/refined_img' % image_path   # Bernardo  (for 2D images)
    img_path = '%s/refined_img' % image_path   # Bernardo  (for 3D point clouds)

    # img_list_path = '%s/meta/%s_name_5pts_score.txt' % (image_path, target.lower())     # original
    img_list_path = '%s/meta/%s_name_5pts_score.txt' % (protocols_path, target.lower())   # Bernardo
    img_list = open(img_list_path)
    files = img_list.readlines()
    # files_list = divideIntoNstrand(files, rank_size)
    files_list = files

    # print('img_path:', img_path)
    # print('files_list:', files_list)
    # sys.exit(0)

    # img_feats
    # for i in range(rank_size):
    img_feats, faceness_scores = get_image_feature(img_path, files_list,
                                                model_path, 0, gpu_id)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))
    print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0],
                                            img_feats.shape[1]))



    # # Step3: Get Template Features

    # In[ ]:

    # =============================================================
    # compute template features from image features.
    # =============================================================
    start = timeit.default_timer()
    # ==========================================================
    # Norm feature before aggregation into template feature?
    # Feature norm from embedding network and faceness score are able to decrease weights for noise samples (not face).
    # ==========================================================
    # 1. FaceScore (Feature Norm)
    # 2. FaceScore (Detector)

    if use_flip_test:
        # concat --- F1
        # img_input_feats = img_feats
        # add --- F2
        img_input_feats = img_feats[:, 0:img_feats.shape[1] //
                                        2] + img_feats[:, img_feats.shape[1] // 2:]
    else:
        img_input_feats = img_feats[:, 0:img_feats.shape[1] // 2]

    if use_norm_score:
        img_input_feats = img_input_feats
    else:
        # normalise features to remove norm information
        img_input_feats = img_input_feats / np.sqrt(
            np.sum(img_input_feats ** 2, -1, keepdims=True))

    if use_detector_score:
        print(img_input_feats.shape, faceness_scores.shape)
        img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]
    else:
        img_input_feats = img_input_feats

    template_norm_feats, unique_templates = image2template_feature(
        img_input_feats, templates, medias)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))




    # # Step 4: Get Template Similarity Scores

    # In[ ]:

    # =============================================================
    # compute verification scores between template pairs.
    # =============================================================
    start = timeit.default_timer()
    score = verification(template_norm_feats, unique_templates, p1, p2)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))


    # In[ ]:
    # exper_id = model_path.split('/')[-2]            # Bernardo
    # save_path = os.path.join(result_dir, exper_id)  # Bernardo
    # save_path = os.path.join(result_dir, args.job)
    # save_path = result_dir + '/%s_result' % target

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # score_save_file = os.path.join(save_path, "%s.npy" % target.lower())
    np.save(score_save_file, score)
    np.save(label_save_file, label)


else:  # Bernardo: load saved scores (distances)
    score = np.load(score_save_file)
    label = np.load(label_save_file)




# # Step 5: Get ROC Curves and TPR@FPR Table

# In[ ]:

files = [score_save_file]
methods = []
scores = []
for file in files:
    methods.append(Path(file).stem)
    scores.append(np.load(file))

methods = np.array(methods)
scores = dict(zip(methods, scores))
colours = dict(
    zip(methods, sample_colours_from_colourmap(methods.shape[0], 'Set2')))
x_labels = [10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]
tpr_fpr_table = PrettyTable(['Methods'] + [str(x) for x in x_labels])
fig = plt.figure()
roc_auc = 0.0
for method in methods:
    fpr, tpr, _ = roc_curve(label, scores[method])
    roc_auc = auc(fpr, tpr)
    fpr = np.flipud(fpr)
    tpr = np.flipud(tpr)  # select largest tpr at same fpr
    plt.plot(fpr,
             tpr,
             color=colours[method],
             lw=1,
             label=('[%s (AUC = %0.4f %%)]' %
                    (method.split('-')[-1], roc_auc * 100)))
    tpr_fpr_row = []
    tpr_fpr_row.append("%s-%s" % (method, target))
    for fpr_iter in np.arange(len(x_labels)):
        _, min_index = min(
            list(zip(abs(fpr - x_labels[fpr_iter]), range(len(fpr)))))
        tpr_fpr_row.append('%.2f' % (tpr[min_index] * 100))
    tpr_fpr_table.add_row(tpr_fpr_row)
plt.xlim([10 ** -6, 0.1])
plt.ylim([0.3, 1.0])
plt.grid(linestyle='--', linewidth=1)
plt.xticks(x_labels)
plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
plt.xscale('log')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC on IJB')
plt.legend(loc="lower right")
fig.savefig(os.path.join(save_path, '%s.pdf' % target.lower()))
print(tpr_fpr_table)

# Bernardo
print('AUC = %0.4f %%' % (roc_auc * 100))









def compute_all_embeddings_and_distances_pointnet2(sess, ops):
    is_training = False

    cur_batch_data = np.zeros((2,BATCH_SIZE,NUM_POINT,EVAL_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    all_distances = np.zeros((0), dtype=np.float32)
    all_pairs_labels = np.zeros((0), dtype=np.int32)
    print('all_distances.shape:', all_distances.shape, end='\r')
    
    while EVAL_DATASET.has_next_batch():
        batch_data, batch_label = EVAL_DATASET.next_batch(augment=False)
        # bsize = batch_data.shape[0]  # original
        bsize = batch_data.shape[1]    # Bernardo
        # for the last batch in the epoch, the bsize:end are from last batch
        
        cur_batch_data[0,0:bsize,...] = batch_data[0]
        cur_batch_data[1,0:bsize,...] = batch_data[1]
        cur_batch_label[0:bsize] = batch_label

        feed_dict0 = {ops['pointclouds_pl']: cur_batch_data[0],
                      ops['labels_pl']: cur_batch_label,
                      ops['is_training_pl']: is_training}
        
        feed_dict1 = {ops['pointclouds_pl']: cur_batch_data[1],
                      ops['labels_pl']: cur_batch_label,
                      ops['is_training_pl']: is_training}

        # loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
        # loss_val, ind_loss, distances, pred_labels = sess.run([ops['total_loss'], ops['individual_losses'], ops['distances'], ops['pred_labels']], feed_dict=feed_dict)
        # batch_pred_sum += pred_labels

        embds0, pred0 = sess.run([ops['embds'], ops['pred']], feed_dict=feed_dict0)
        embds1, pred1 = sess.run([ops['embds'], ops['pred']], feed_dict=feed_dict1)
        # pred_labels0 = np.argmax(pred_val0, 1)
        # pred_labels1 = np.argmax(pred_val1, 1)
        
        distances = cosine_distance(embds0, embds1)
        distances = distances[0:bsize]

        all_distances = np.append(all_distances, distances, axis=0)
        all_pairs_labels = np.append(all_pairs_labels, cur_batch_label[0:bsize], axis=0)
        print('all_distances.shape:', all_distances.shape, end='\r')
    print()

    EVAL_DATASET.reset()
    return all_distances, all_pairs_labels







'''
start = timeit.default_timer()
# img_path = '%s/loose_crop' % image_path
img_path = '%s/refined_img' % image_path
img_list_path = '%s/meta/%s_name_5pts_score.txt' % (image_path, target.lower())
img_list = open(img_list_path)
files = img_list.readlines()
# files_list = divideIntoNstrand(files, rank_size)
files_list = files

# evaluate_varying_margin(num_votes=FLAGS.num_votes)
img_feats, faceness_scores = get_image_feature(image_path, files_list, model_path, 0, gpu_id)
'''
