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
import scipy.misc
from scipy import interpolate
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, '../models'))
sys.path.append(os.path.join(ROOT_DIR, '../utils'))
import provider
# import modelnet_dataset
# import modelnet_h5_dataset

from data_loader.loader_reconstructed_MICA import lfw_evaluation_3Dreconstructed_MICA_dataset_pairs      # Bernardo
from data_loader.loader_reconstructed_MICA import magVerif_pairs_3Dreconstructed_MICA                    # Bernardo
from data_loader.loader_reconstructed_MICA import calfw_evaluation_3Dreconstructed_MICA_dataset_pairs    # Bernardo
from data_loader.loader_reconstructed_HRN import magVerif_pairs_3Dreconstructed_HRN

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
# parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name [default: pointnet2_cls_ssg]')  # original
parser.add_argument('--model', default='pointnet2_cls_ssg_angmargin', help='Model name [default: pointnet2_cls_ssg_angmargin]')    # Bernardo
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
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
parser.add_argument('--margin', type=float, default=0.1, help='Minimum distance for non-corresponding pairs in Contrastive Loss')

# parser.add_argument('--dataset', type=str, default='frgc', help='Name of dataset to train model')                 # Bernardo
# parser.add_argument('--dataset', type=str, default='synthetic_gpmm', help='Name of dataset to train model')       # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_ms1mv2', help='Name of dataset to train model')  # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_lfw', help='Name of dataset to train model')     # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_agedb', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_cfp', help='Name of dataset to train model')     # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_calfw', help='Name of dataset to train model')   # Bernardo
parser.add_argument('--dataset', type=str, default='reconst_hrn_lfw', help='Name of dataset to train model')        # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_hrn_agedb', help='Name of dataset to train model')    # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_hrn_cfp', help='Name of dataset to train model')      # Bernardo

# Only for fusion of 2D and 3D models
parser.add_argument('--fusion', action='store_true')
parser.add_argument('--arc_dists', type=str, default='/home/bjgbiesseck/GitHub/InsightFace-tensorflow/output/dataset=MS1MV3_1000subj_classes=1000_backbone=resnet-v2-m-50_epoch-num=100_margin=0.5_scale=64.0_lr=0.01_wd=0.0005_momentum=0.9_20230518-004011/lfw_distances_arcface=1000class_acc=0.94650.npy', help='')     # Bernardo

FLAGS = parser.parse_args()



NUM_CLASSES = FLAGS.num_class
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
MARGIN = FLAGS.margin
ARCFACE_DISTANCES_FILE = FLAGS.arc_dists


MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

HOSTNAME = socket.gethostname()


print('\nLoading dataset paths...')
if FLAGS.dataset.upper() == 'reconst_mica_lfw'.upper():
    DATA_PATH = os.path.join(ROOT_DIR, '../../BOVIFOCR_MICA_3Dreconstruction/demo/output/lfw')                    # duo
    # DATA_PATH = os.path.join(ROOT_DIR, '/nobackup/unico/datasets/face_recognition/MICA_3Dreconstruction/lfw')   # diolkos
    # DATA_PATH = os.path.join(ROOT_DIR, '/nobackup1/bjgbiesseck/datasets/MICA_3Dreconstruction/lfw')             # peixoto
    EVAL_DATASET = lfw_evaluation_3Dreconstructed_MICA_dataset_pairs.LFW_Evaluation_3D_Reconstructed_MICA_Dataset_Pairs(root=DATA_PATH, npoints=NUM_POINT, normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

elif FLAGS.dataset.upper() == 'reconst_mica_agedb'.upper():
    # DATA_PATH = os.path.join(ROOT_DIR, '/datasets2/pbqv20/agedb_bkp/agedb_3d')  # duo
    DATA_PATH = os.path.join(ROOT_DIR, '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/lfw_cfp_agedb/agedb/imgs')   # duo
    protocol_file_path = '/datasets2/pbqv20/agedb_bkp/pairs.txt'                  # duo
    EVAL_DATASET = magVerif_pairs_3Dreconstructed_MICA.MAGFACE_Evaluation_3D_Reconstructed_MICA_Dataset_Pairs(root=DATA_PATH, protocol_file_path=protocol_file_path, npoints=NUM_POINT, normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

elif FLAGS.dataset.upper() == 'reconst_mica_cfp'.upper():
    # DATA_PATH = os.path.join(ROOT_DIR, '/datasets2/pbqv20/cfp_bkp/cfp_3d')  # duo
    DATA_PATH = os.path.join(ROOT_DIR, '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/lfw_cfp_agedb/cfp/imgs')   # duo
    protocol_file_path = '/datasets2/pbqv20/cfp_bkp/pairs.txt'                # duo
    EVAL_DATASET = magVerif_pairs_3Dreconstructed_MICA.MAGFACE_Evaluation_3D_Reconstructed_MICA_Dataset_Pairs(root=DATA_PATH, protocol_file_path=protocol_file_path, npoints=NUM_POINT, normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

elif FLAGS.dataset.upper() == 'reconst_mica_calfw'.upper():
    DATA_PATH = os.path.join(ROOT_DIR, '../../BOVIFOCR_MICA_3Dreconstruction/demo/output/calfw')
    EVAL_DATASET = calfw_evaluation_3Dreconstructed_MICA_dataset_pairs.CALFW_Evaluation_3D_Reconstructed_MICA_Dataset_Pairs(root=DATA_PATH, npoints=NUM_POINT, normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

elif FLAGS.dataset.upper() == 'reconst_hrn_lfw'.upper():
    DATA_PATH = '/datasets1/bjgbiesseck/lfw_cfp_agedb/3D_reconstruction_HRN/lfw/imgs'                 # duo
    protocol_file_path = '/datasets1/bjgbiesseck/lfw_cfp_agedb/rgb/lfw/pair.list'                     # duo
    EVAL_DATASET = magVerif_pairs_3Dreconstructed_HRN.MAGFACE_Evaluation_3D_Reconstructed_HRN_Dataset_Pairs(root=DATA_PATH, protocol_file_path=protocol_file_path, npoints=NUM_POINT, normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

elif FLAGS.dataset.upper() == 'reconst_hrn_agedb'.upper():
    DATA_PATH = '/datasets1/bjgbiesseck/lfw_cfp_agedb/3D_reconstruction_HRN/agedb/imgs'               # duo
    protocol_file_path = '/datasets1/bjgbiesseck/lfw_cfp_agedb/rgb/agedb/pair.list'                   # duo
    EVAL_DATASET = magVerif_pairs_3Dreconstructed_HRN.MAGFACE_Evaluation_3D_Reconstructed_HRN_Dataset_Pairs(root=DATA_PATH, protocol_file_path=protocol_file_path, npoints=NUM_POINT, normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)

elif FLAGS.dataset.upper() == 'reconst_hrn_cfp'.upper():
    DATA_PATH = '/datasets1/bjgbiesseck/lfw_cfp_agedb/3D_reconstruction_HRN/cfp/imgs'                 # duo
    protocol_file_path = '/datasets1/bjgbiesseck/lfw_cfp_agedb/rgb/cfp/pair.list'                     # duo
    EVAL_DATASET = magVerif_pairs_3Dreconstructed_HRN.MAGFACE_Evaluation_3D_Reconstructed_HRN_Dataset_Pairs(root=DATA_PATH, protocol_file_path=protocol_file_path, npoints=NUM_POINT, normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)





class LFold:
    def __init__(self, n_splits=2, shuffle=False):
        self.n_splits = n_splits
        if self.n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        else:
            return [(indices, indices)]




def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


# Bernardo
def save_metric_to_text_file(path_file, all_margins_eval, all_tp_eval, all_fp_eval, all_tn_eval, all_fn_eval, all_acc_eval, all_far_eval, all_tar_eval):
    with open(path_file, 'w') as f:
        f.write('margin,tp,fp,tn,fn,acc,far,tar' + '\n')
        for margin, tp, fp, tn, fn, acc, far, tar in zip(all_margins_eval, all_tp_eval, all_fp_eval, all_tn_eval, all_fn_eval, all_acc_eval, all_far_eval, all_tar_eval):
            f.write(str(margin) + ',' + str(tp) + ',' + str(fp) + ',' + str(tn) + ',' + str(fn) + ',' + str(acc) + ',' + str(far) + ',' + str(tar) + '\n')
            f.flush()


def cosine_distance_insightface(embds0, embds1):
    assert embds0.shape[0] == embds1.shape[0], f'Error, sizes of embds0 ({embds0.shape[0]}) and embds1 ({embds1.shape[0]}) are different. Must be equal!\n'
    embds0 = normalize(embds0)
    embds1 = normalize(embds1)
    distances = np.sum(np.square(embds0 - embds1), axis=1)
    return distances

def cosine_distance(embds0, embds1):
    assert embds0.shape[0] == embds1.shape[0], f'Error, sizes of embds0 ({embds0.shape[0]}) and embds1 ({embds1.shape[0]}) are different. Must be equal!\n'
    cos_dist = np.zeros(embds0.shape[0], dtype=np.float32)
    for i in range(cos_dist.shape[0]):
        cos_dist[i] = 1.0 - np.dot(embds0[i], embds1[i])/(np.linalg.norm(embds0[i])*np.linalg.norm(embds1[i]))
    return cos_dist


# Bernardo
def compute_all_embeddings_and_distances_pointnet2(sess, ops):
    is_training = False

    cur_batch_data = np.zeros((2,BATCH_SIZE,NUM_POINT,EVAL_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    all_distances = np.zeros((len(EVAL_DATASET)), dtype=np.float32)
    all_pairs_labels = np.zeros((len(EVAL_DATASET)), dtype=np.int32)
    # print('all_distances.shape:', all_distances.shape, end='\r')
    
    curr_idx = 0
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

        embd0, logits0 = sess.run([ops['embd'], ops['logits']], feed_dict=feed_dict0)
        embd1, logits1 = sess.run([ops['embd'], ops['logits']], feed_dict=feed_dict1)
        
        # distances = cosine_distance(embd0, embd1)
        distances = cosine_distance_insightface(embd0, embd1)
        distances = distances[0:bsize]

        # all_distances = np.append(all_distances, distances, axis=0)
        # all_pairs_labels = np.append(all_pairs_labels, cur_batch_label[0:bsize], axis=0)
        all_distances[curr_idx:curr_idx+bsize] = distances
        all_pairs_labels[curr_idx:curr_idx+bsize] = cur_batch_label[0:bsize]
        print(f'all_distances: {curr_idx}/{len(EVAL_DATASET)}', end='\r')

        curr_idx += bsize
        
    print()

    EVAL_DATASET.reset()
    return all_distances, all_pairs_labels


def get_tp_fp_tn_fn_pairs_indexes(predict_issame, actual_issame):
    tp_idx = np.logical_and(predict_issame, actual_issame)
    fp_idx = np.logical_and(predict_issame, np.logical_not(actual_issame))
    tn_idx = np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame))
    fn_idx = np.logical_and(np.logical_not(predict_issame), actual_issame)

    tp_idx = np.where(tp_idx == True)[0]
    fp_idx = np.where(fp_idx == True)[0]
    tn_idx = np.where(tn_idx == True)[0]
    fn_idx = np.where(fn_idx == True)[0]

    return tp_idx, fp_idx, tn_idx, fn_idx


def get_true_accept_false_accept_pairs_indexes(predict_issame, actual_issame):
    ta_idx = np.logical_and(predict_issame, actual_issame)
    fa_idx = np.logical_and(predict_issame, np.logical_not(actual_issame))

    ta_idx = np.where(ta_idx == True)[0]
    fa_idx = np.where(fa_idx == True)[0]

    return ta_idx, fa_idx


def calculate_accuracy_tp_fp_tn_fn_pairs_indexes(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)

    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tp_idx, fp_idx, tn_idx, fn_idx = get_tp_fp_tn_fn_pairs_indexes(predict_issame, actual_issame)

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)

    # acc = float(tp + tn) / dist.size
    acc = float(tp + tn) / actual_issame.size
    return tpr, fpr, acc, tp_idx, fp_idx, tn_idx, fn_idx


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(
        np.logical_and(np.logical_not(predict_issame),
                    np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_roc(thresholds, dist, actual_issame, nrof_folds=10, verbose=True):
        # assert (embeddings1.shape[0] == embeddings2.shape[0])
        # assert (embeddings1.shape[1] == embeddings2.shape[1])
        assert (dist.shape[0] == actual_issame.shape[0])   # Bernardo
        nrof_pairs = min(len(actual_issame), dist.shape[0])
        nrof_thresholds = len(thresholds)
        k_fold = LFold(n_splits=nrof_folds, shuffle=False)

        tprs = np.zeros((nrof_folds, nrof_thresholds))
        fprs = np.zeros((nrof_folds, nrof_thresholds))
        accuracy = np.zeros((nrof_folds))
        indices = np.arange(nrof_pairs)

        tp_idx = [None] * nrof_folds
        fp_idx = [None] * nrof_folds
        tn_idx = [None] * nrof_folds
        fn_idx = [None] * nrof_folds

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            if verbose:
                # print(f'calculate_roc - fold_idx: {fold_idx}/{nrof_folds-1}', end='\r')
                print('calculate_roc - fold_idx: '+str(fold_idx)+'/'+str(nrof_folds-1), end='\r')

            # Find the best threshold for the fold
            acc_train = np.zeros((nrof_thresholds))
            for threshold_idx, threshold in enumerate(thresholds):            
                _, _, acc_train[threshold_idx] = calculate_accuracy(
                    threshold, dist[train_set], actual_issame[train_set])
            best_threshold_index = np.argmax(acc_train)
            for threshold_idx, threshold in enumerate(thresholds):
                tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                    threshold, dist[test_set],
                    actual_issame[test_set])

            # original
            # _, _, accuracy[fold_idx] = self.calculate_accuracy(
            #     thresholds[best_threshold_index], dist[test_set],
            #     actual_issame[test_set])

            # Bernardo
            _, _, accuracy[fold_idx], tp_idx[fold_idx], fp_idx[fold_idx], tn_idx[fold_idx], fn_idx[fold_idx] = calculate_accuracy_tp_fp_tn_fn_pairs_indexes(
                thresholds[best_threshold_index], dist[test_set],
                actual_issame[test_set])

            tp_idx[fold_idx] = test_set[tp_idx[fold_idx]]
            fp_idx[fold_idx] = test_set[fp_idx[fold_idx]]
            tn_idx[fold_idx] = test_set[tn_idx[fold_idx]]
            fn_idx[fold_idx] = test_set[fn_idx[fold_idx]]

        tp_idx = np.concatenate(tp_idx)
        fp_idx = np.concatenate(fp_idx)
        tn_idx = np.concatenate(tn_idx)
        fn_idx = np.concatenate(fn_idx)

        if verbose:
            print('')

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
        # return tpr, fpr, accuracy
        return tpr, fpr, accuracy, tp_idx, fp_idx, tn_idx, fn_idx


def calculate_tar_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(
        np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    tar = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return tar, far


def calculate_tar_far_tp_fp_tn_fn_pairs_indexes(threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)

        true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
        false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        n_same = np.sum(actual_issame)
        n_diff = np.sum(np.logical_not(actual_issame))

        ta_idx, fa_idx = get_true_accept_false_accept_pairs_indexes(predict_issame, actual_issame)

        tar = float(true_accept) / float(n_same)
        far = float(false_accept) / float(n_diff)
        return tar, far, ta_idx, fa_idx


def calculate_tar(thresholds, dist, actual_issame, far_target, nrof_folds=10, verbose=True):
        # assert (embeddings1.shape[0] == embeddings2.shape[0])
        # assert (embeddings1.shape[1] == embeddings2.shape[1])
        nrof_pairs = min(len(actual_issame), dist.shape[0])
        nrof_thresholds = len(thresholds)
        k_fold = LFold(n_splits=nrof_folds, shuffle=False)

        tar = np.zeros(nrof_folds)
        far = np.zeros(nrof_folds)

        # diff = np.subtract(embeddings1, embeddings2)
        # dist = np.sum(np.square(diff), 1)
        indices = np.arange(nrof_pairs)

        ta_idx = [None] * nrof_folds
        fa_idx = [None] * nrof_folds

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            if verbose:
                # print(f'calculate_tar - fold_idx: {fold_idx}/{nrof_folds-1}', end='\r')
                print('calculate_tar - fold_idx: '+str(fold_idx)+'/'+str(nrof_folds-1), end='\r')

            # Find the threshold that gives FAR = far_target
            far_train = np.zeros(nrof_thresholds)
            for threshold_idx, threshold in enumerate(thresholds):
                _, far_train[threshold_idx] = calculate_tar_far(
                    threshold, dist[train_set], actual_issame[train_set])
            if np.max(far_train) >= far_target:
                f = interpolate.interp1d(far_train, thresholds, kind='slinear')
                threshold = f(far_target)
            else:
                threshold = 0.0

            # original
            # tar[fold_idx], far[fold_idx] = self.calculate_tar_far(
            #     threshold, dist[test_set], actual_issame[test_set])

            # Bernardo
            tar[fold_idx], far[fold_idx], ta_idx[fold_idx], fa_idx[fold_idx] = calculate_tar_far_tp_fp_tn_fn_pairs_indexes(
                threshold, dist[test_set], actual_issame[test_set])

            ta_idx[fold_idx] = test_set[ta_idx[fold_idx]]
            fa_idx[fold_idx] = test_set[fa_idx[fold_idx]]

        ta_idx = np.concatenate(ta_idx)
        fa_idx = np.concatenate(fa_idx)

        if verbose:
            print('')

        tar_mean = np.mean(tar)
        far_mean = np.mean(far)
        tar_std = np.std(tar)
        # return tar_mean, tar_std, far_mean
        return tar_mean, tar_std, far_mean, ta_idx, fa_idx


def do_k_fold_test(folds_pair_distances, folds_pair_labels, verbose=True):
        thresholds = np.arange(0, 4, 0.01)
        # tpr, fpr, accuracy = self.calculate_roc(thresholds, folds_pair_distances, folds_pair_labels, nrof_folds=10, verbose=verbose)
        tpr, fpr, accuracy, tp_idx, fp_idx, tn_idx, fn_idx = calculate_roc(thresholds, folds_pair_distances, folds_pair_labels, nrof_folds=10, verbose=verbose)
        # print('tp_idx.shape:', tp_idx.shape)
        # print('fp_idx.shape:', fp_idx.shape)
        # print('tn_idx.shape:', tn_idx.shape)
        # print('fn_idx.shape:', fn_idx.shape)

        thresholds = np.arange(0, 4, 0.001)
        # tar_mean, tar_std, far_mean = self.calculate_tar(thresholds, folds_pair_distances, folds_pair_labels, far_target=1e-3, nrof_folds=10, verbose=verbose)
        tar_mean, tar_std, far_mean, ta_idx, fa_idx = calculate_tar(thresholds, folds_pair_distances, folds_pair_labels, far_target=1e-3, nrof_folds=10, verbose=verbose)
        # print('ta_idx.shape:', ta_idx.shape)
        # print('fa_idx.shape:', fa_idx.shape)

        if verbose:
            print('------------')

        # return tpr, fpr, accuracy, tar_mean, tar_std, far_mean
        return tpr, fpr, accuracy, tar_mean, tar_std, far_mean, \
            tp_idx, fp_idx, tn_idx, fn_idx, ta_idx, fa_idx


# Bernardo
def evaluate_varying_margin(num_votes):
    is_training = False

    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        interm_layers, embd, logits, end_points, weights_fc3 = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=None, num_class=NUM_CLASSES)    # Bernardo
        logits, loss, classify_loss = MODEL.get_loss_arcface(logits, labels_pl, end_points, weights_fc3, num_classes=NUM_CLASSES)

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
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'embd': embd,
           'logits': logits,
           'loss': total_loss,
           'end_points': end_points}


    print('\nPointNet++ (3D) - Computing all embeddings and distances...')
    all_distances_pointnet2, pairs_labels = compute_all_embeddings_and_distances_pointnet2(sess, ops)
    # print('all_distances_pointnet2.shape:', all_distances_pointnet2.shape)
    # print('pairs_labels:', pairs_labels)
    # print('pairs_labels.shape:', pairs_labels.shape)
    # sys.exit(0)

    if FLAGS.fusion and FLAGS.arc_dists != '':
        print(f'\nLoading Arcface (2D) pair distances: \'{FLAGS.arc_dists}\'')
        all_distances_arcface = np.load(FLAGS.arc_dists)
        print('')

        assert all_distances_pointnet2.shape[0] == all_distances_arcface.shape[0], f'Error, all_distances_pointnet2.shape[0] ({all_distances_pointnet2.shape[0]}) and all_distances_arcface.shape[0] ({all_distances_arcface.shape[0]}) must be equal'
        print(f'Fusing scores 2D and 3D...')
        all_distances_pointnet2 = (all_distances_pointnet2 + all_distances_arcface) / 2.0
        print('')


    tpr, fpr, accuracy, tar_mean, tar_std, far_mean, \
            tp_idx, fp_idx, tn_idx, fn_idx, ta_idx, fa_idx = do_k_fold_test(all_distances_pointnet2, pairs_labels, verbose=True)
    acc_mean, acc_std = np.mean(accuracy), np.std(accuracy)

    print('\nMODEL_PATH:', MODEL_PATH)
    print('Final - dataset: %s  -  acc_mean: %.6f +- %.6f  -  tar: %.6f +- %.6f    far: %.6f' % (FLAGS.dataset, acc_mean, acc_std, tar_mean, tar_std, far_mean))
    print('Finished!')



if __name__=='__main__':
    with tf.Graph().as_default():
        # evaluate(num_votes=FLAGS.num_votes)
        evaluate_varying_margin(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()
