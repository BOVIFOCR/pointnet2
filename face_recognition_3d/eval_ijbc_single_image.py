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


parser = argparse.ArgumentParser(description='do ijb test')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
# parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name [default: pointnet2_cls_ssg]')  # original
parser.add_argument('--model', default='pointnet2_cls_ssg_angmargin', help='Model name [default: pointnet2_cls_ssg_angmargin]')    # Bernardo
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size during training [default: 16]')
# parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')    # original
parser.add_argument('--num_point', type=int, default=2900, help='Point Number [default: 1024]')      # Bernardo

# parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')   # original
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
image_path = '/datasets1/bjgbiesseck/IJB-C/IJB/IJB-C/3D_reconstruction_MICA/crops_align'
result_dir = 'results_ijbc_single_img'
gpu_id = None
use_norm_score = True  # if Ture, TestMode(N1)
# use_detector_score = True  # if Ture, TestMode(D1)
use_detector_score = False   # if Ture, TestMode(D1)
# use_flip_test = True  # if Ture, TestMode(F1)
use_flip_test = False   # if Ture, TestMode(F1)
job = 'insightface'
# batch_size = 128


class Embedding(object):
    def __init__(self, prefix, data_shape, batch_size=1):
        image_size = (112, 112)
        self.image_size = image_size

        # LOAD TRAINED MODEL (original)
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
            print('\nError, nan found in input data')
            print('batch_data:', batch_data)
            print('batch_data.shape:', batch_data.shape)
            sys.exit(0)
        
        if np.sum(np.isnan(cur_embd)) > 0:
            print('\nError, nan found in output embeddings')
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


# BERNARDO
def read_template_original_ijbc(path):
    # ijb_meta = np.loadtxt(path, dtype=str)
    ijb_meta = pd.read_csv(path, sep=',', header=0).values
    template_id = ijb_meta[:, 0].astype(int)
    subject_id = ijb_meta[:, 1]
    filename = ijb_meta[:, 2]
    return template_id, subject_id, filename


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


# BERNARDO
def read_template_pair_list_original_ijbc(path):
    pairs = pd.read_csv(path, sep=',', header=0).values
    t1 = pairs[:, 0].astype(np.int)
    t2 = pairs[:, 1].astype(np.int)
    # label = pairs[:, 2].astype(np.int)
    # return t1, t2, label
    return t1, t2


# BERNARDO
def make_labels_from_template_pairs_original_ijbc(enroll_template_id, enroll_subject_id, verif_template_id, verif_subject_id, p1, p2):
    assert len(p1) == len(p2)

    enroll_template_dict = {}
    for i in range(len(enroll_template_id)):
        enroll_template_dict[enroll_template_id[i]] = enroll_subject_id[i]

    verif_template_dict = {}
    for i in range(len(verif_template_id)):
        verif_template_dict[verif_template_id[i]] = verif_subject_id[i]

    label = np.zeros((len(p1)), dtype=int)
    for i, (t1, t2) in enumerate(zip(p1, p2)):
        label[i] = int(enroll_template_dict[t1] == verif_template_dict[t2])

    return label


# BERNARDO
def adjust_file_names(enroll_subject_id, enroll_filenames, verif_subject_id, verif_filenames):
    enroll_img_paths = [None] * len(enroll_filenames)
    verif_img_paths = [None] * len(verif_filenames)

    for i in range(len(enroll_img_paths)):
        enroll_data = enroll_filenames[i].split('/')
        enroll_img_paths[i] = os.path.join(enroll_data[0], str(enroll_subject_id[i])+'_'+enroll_data[1].replace('.png', '.jpg')) + str(' -1' * 10) + ' 1.0'

    for i in range(len(verif_filenames)):
        verif_data = verif_filenames[i].split('/')
        verif_img_paths[i] = os.path.join(verif_data[0], str(verif_subject_id[i])+'_'+verif_data[1].replace('.png', '.jpg')) + str(' -1' * 10) + ' 1.0'

    files_list = enroll_img_paths
    return files_list



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

    # LOAD TRAINED MODEL (Bernardo)
    is_training = False
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        embd, end_points, weights_fc3 = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=None, num_class=NUM_CLASSES)    # Bernardo
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
        batch_data = np.empty((batch_size, 2900, 3))            # Bernardo
        embedding = Embedding(model_path, data_shape, batch_size)
        num_batches = int(np.ceil(len(files) / batch_size))
        print('Computing embeddings...')
        for img_index, each_line in enumerate(files[:len(files) - rare_size]):
            name_lmk_score = each_line.strip().split(' ')
            
            # img_name = os.path.join(img_path, name_lmk_score[0])                    # original
            file_ext = 'mesh_centralized-nosetip_with-normals_filter-radius=100.npy'  # Bernardo
            folder_name = name_lmk_score[0].split('.')[0]                             # Bernardo
            img_name = os.path.join(img_path, folder_name, file_ext)                  # Bernardo
            # print('get_image_feature - img_name:', img_name)

            '''  # original (not necessary for 3D point clouds)
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

            # batch_data[2 * (img_index - batch * batch_size)][:] = input_blob[0]       # original
            # batch_data[2 * (img_index - batch * batch_size) + 1][:] = input_blob[1]   # original
            batch_data[(img_index - batch * batch_size)][:] = point_set[:2900]          # Bernardo
            if (img_index + 1) % batch_size == 0:
                # print('batch', batch)
                sys.stdout.write('batch {}/{}\r'.format(batch, num_batches-1))
                sys.stdout.flush()
                img_feats[batch * batch_size:batch * batch_size +
                                            batch_size][:] = embedding.forward_db(batch_data, sess, ops)
                batch += 1
            faceness_scores.append(name_lmk_score[-1])

        # batch_data = np.empty((2 * rare_size, 3, 112, 112))   # original
        batch_data = np.empty((rare_size, 2900, 3))             # Bernardo
        embedding = Embedding(model_path, data_shape, rare_size)
        for img_index, each_line in enumerate(files[len(files) - rare_size:]):
            name_lmk_score = each_line.strip().split(' ')

            # img_name = os.path.join(img_path, name_lmk_score[0])                    # original
            file_ext = 'mesh_centralized-nosetip_with-normals_filter-radius=100.npy'  # Bernardo
            folder_name = name_lmk_score[0].split('.')[0]                             # Bernardo
            img_name = os.path.join(img_path, folder_name, file_ext)                  # Bernardo
            # print('get_image_feature - img_name:', img_name)

            '''  # original (not necessary for 3D point clouds)
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

            # batch_data[2 * img_index][:] = input_blob[0]       # original
            # batch_data[2 * img_index + 1][:] = input_blob[1]   # original
            batch_data[img_index][:] = point_set[:2900]          # Bernardo
            if (img_index + 1) % rare_size == 0:
                # print('batch', batch)
                sys.stdout.write('batch {}/{}\r'.format(batch, num_batches-1))
                sys.stdout.flush()
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
            # print('Finish Calculating {} template features.'.format(count_template))
            sys.stdout.write('Finish Calculating template features: {}/{}\r'.format(count_template, img_feats.shape[0]))
            sys.stdout.flush()
    # template_norm_feats = template_feats / np.sqrt(np.sum(template_feats ** 2, -1, keepdims=True))
    template_norm_feats = sklearn.preprocessing.normalize(template_feats)
    print('')
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
            # print('Finish {}/{} pairs.'.format(c, total_sublists))
            sys.stdout.write('Finish {}/{} pairs.\r'.format(c, total_sublists))
            sys.stdout.flush()
    print('')
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
img_feats_save_file = os.path.join(save_path, "img_feats.npy")
faceness_scores_save_file = os.path.join(save_path, "faceness_scores.npy")

# Verification protocol files
path_enroll_templates = '/datasets1/bjgbiesseck/IJB-C/IJB/IJB-C/protocols/test2/enroll_templates.csv'
path_verif_templates =  '/datasets1/bjgbiesseck/IJB-C/IJB/IJB-C/protocols/test2/verif_templates.csv'
path_match_pairs =      '/datasets1/bjgbiesseck/IJB-C/IJB/IJB-C/protocols/test2/match.csv'
# path_match_pairs =    '/datasets1/bjgbiesseck/IJB-C/IJB/IJB-C/protocols/test2/match_TEST_BERNARDO.csv'  # Toy example for sanity check



# # Step1: Load Meta Data
# In[ ]:
# =============================================================
# load image and template relationships for template feature embedding
# tid --> template id,  mid --> media id
# format:
#           image_name tid mid
# =============================================================
assert target == 'IJBC' or target == 'IJBB'
start = timeit.default_timer()
# templates, medias = read_template_media_list(os.path.join('%s/meta' % image_path, '%s_face_tid_mid.txt' % target.lower()))
print('Loading enroll templates \'{}\'...'.format(path_enroll_templates))
enroll_template_id, enroll_subject_id, enroll_filenames = read_template_original_ijbc(path_enroll_templates)  # one image protocol
print('num enroll templates: {}'.format(len(enroll_template_id)))
print('Loading verification templates \'{}\'...'.format(path_verif_templates))
verif_template_id, verif_subject_id, verif_filenames = read_template_original_ijbc(path_verif_templates)      # one image protocol
print('num verif templates: {}'.format(len(verif_template_id)))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('----------------------')

# In[ ]:
# =============================================================
# load template pairs for template-to-template verification
# tid : template id,  label : 1/0
# format:
#           tid_1 tid_2 label
# =============================================================
start = timeit.default_timer()
# p1, p2, label = read_template_pair_list(os.path.join('%s/meta' % image_path, '%s_template_pair_label.txt' % target.lower()))
print('Loading match pairs indexes \'{}\'...'.format(path_match_pairs))
p1, p2 = read_template_pair_list_original_ijbc(path_match_pairs)
label = make_labels_from_template_pairs_original_ijbc(enroll_template_id, enroll_subject_id, verif_template_id, verif_subject_id, p1, p2)
num_genuine_pairs = np.sum(label == 1)
num_impostor_pairs = np.sum(label == 0)
assert num_genuine_pairs+num_impostor_pairs == len(label)
print('num pairs: {}    (genuine: {}    impostor: {})'.format(len(label), num_genuine_pairs, num_impostor_pairs))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('----------------------')





# # Step 2: Get Image Features
if not os.path.exists(img_feats_save_file):   # Bernardo
    # In[ ]:
    # =============================================================
    # load image features
    # format:
    #           img_feats: [image_num x feats_dim] (227630, 512)
    # =============================================================
    start = timeit.default_timer()
    img_path = image_path     # Bernardo
    files_list = adjust_file_names(enroll_subject_id, enroll_filenames, verif_subject_id, verif_filenames)   # Bernardo

    img_feats, faceness_scores = get_image_feature(img_path, files_list, model_path, 0, gpu_id)
    stop = timeit.default_timer()
    print('Time: %.2f s. ' % (stop - start))
    print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0], img_feats.shape[1]))
    
    # Bernardo
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('Saving img_feats:', img_feats_save_file)
    np.save(img_feats_save_file, img_feats)
    print('Saving faceness_scores:', img_feats_save_file)
    np.save(faceness_scores_save_file, faceness_scores)

else:
    print('Loading img_feats:', img_feats_save_file)
    img_feats = np.load(img_feats_save_file)
    print('Loading faceness_scores:', img_feats_save_file)
    faceness_scores = np.load(faceness_scores_save_file)

    print('Feature Shape: ({} , {}) .'.format(img_feats.shape[0], img_feats.shape[1]))
print('----------------------')





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
    # img_input_feats = img_feats[:, 0:img_feats.shape[1] // 2]
    img_input_feats = img_feats

if use_norm_score:
    img_input_feats = img_input_feats
else:
    # normalise features to remove norm information
    img_input_feats = img_input_feats / np.sqrt(
        np.sum(img_input_feats ** 2, -1, keepdims=True))

if use_detector_score:
    img_input_feats = img_input_feats * faceness_scores[:, np.newaxis]
else:
    img_input_feats = img_input_feats

print('Computing template embeddings...')
# template_norm_feats, unique_templates = image2template_feature(img_input_feats, templates, medias)                     # original
template_norm_feats, unique_templates = image2template_feature(img_input_feats, enroll_template_id, verif_template_id)   # Bernardo
print('Template Features Shape: ({} , {}) .'.format(template_norm_feats.shape[0], template_norm_feats.shape[1]))
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))
print('----------------------')





# # Step 4: Get Template Similarity Scores
# In[ ]:

# =============================================================
# compute verification scores between template pairs.
# =============================================================
start = timeit.default_timer()
print('Computing pairs similarity scores...')
score = verification(template_norm_feats, unique_templates, p1, p2)
stop = timeit.default_timer()
print('Time: %.2f s. ' % (stop - start))

if not os.path.exists(save_path):
    os.makedirs(save_path)
# score_save_file = os.path.join(save_path, "%s.npy" % target.lower())
print('Saving scores:', score_save_file)
np.save(score_save_file, score)
print('Saving labels:', label_save_file)
np.save(label_save_file, label)
print('----------------------')




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
print('Evaluating model...')
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
# fig.savefig(os.path.join(save_path, '%s.pdf' % target.lower()))
path_save_fig = os.path.join(save_path, '%s.pdf' % target.lower())
print('Saving ROC curve \'{}\''.format(path_save_fig))
fig.savefig(path_save_fig)

path_save_tpr_fpr_table = os.path.join(save_path, '%s_tpr_fpr_table.txt' % target.lower())
with open(path_save_tpr_fpr_table, 'w') as f:
    print('Saving tpr_fpr_table \'{}\''.format(path_save_tpr_fpr_table))
    f.write('tpr_fpr_table\n')
    f.write(tpr_fpr_table.get_string())
    f.write('\nROC-AUC = %0.4f %%\n' % (roc_auc * 100))

print(tpr_fpr_table)
print('ROC-AUC = %0.4f %%' % (roc_auc * 100))   # Bernardo
