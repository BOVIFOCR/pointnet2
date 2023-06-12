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
from data_loader.loader_reconstructed_MICA import calfw_evaluation_3Dreconstructed_MICA_dataset_pairs    # Bernardo

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
# parser.add_argument('--model_path', default='logs_training/classification/dataset=reconst_mica_ms1mv2_10000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=32_m=0.0_26052023_222414/model_best_train_accuracy.ckpt', help='model checkpoint file path')  # Bernardo
# parser.add_argument('--model_path', default='logs_training/classification/dataset=reconst_mica_webface_1000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=1e-05_moment=0.9_loss=arcface_s=32_m=0.0_05062023_170747/model_best_train_accuracy.ckpt', help='model checkpoint file path')  # Bernardo
# parser.add_argument('--model_path', default='logs_training/classification/dataset=reconst_mica_webface_1000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=16_m=0.0_05062023_194932/model_best_train_accuracy.ckpt', help='model checkpoint file path')  # Bernardo
# parser.add_argument('--model_path', default='logs_training/classification/dataset=reconst_mica_webface_2000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=16_m=0.0_05062023_213735/model_best_train_accuracy.ckpt', help='model checkpoint file path')  # Bernardo
parser.add_argument('--model_path', default='logs_training/classification/dataset=reconst_mica_webface_5000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=1e-05_moment=0.9_loss=arcface_s=32_m=0.0_07062023_142326/model_best_train_accuracy.ckpt', help='model checkpoint file path')  # Bernardo


parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
# parser.add_argument('--normal', action='store_true', help='Whether to use normal information')      # original
parser.add_argument('--normal', type=bool, default=False, help='Whether to use normal information')   # Bernardo
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores from multiple rotations [default: 1]')
parser.add_argument('--margin', type=float, default=0.5, help='Minimum distance for non-corresponding pairs in Contrastive Loss')

# parser.add_argument('--dataset', type=str, default='frgc', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='synthetic_gpmm', help='Name of dataset to train model')   # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_ms1mv2', help='Name of dataset to train model')  # Bernardo
parser.add_argument('--dataset', type=str, default='reconst_mica_lfw', help='Name of dataset to train model')       # Bernardo
# parser.add_argument('--dataset', type=str, default='reconst_mica_calfw', help='Name of dataset to train model')   # Bernardo

FLAGS = parser.parse_args()


# NUM_CLASSES = 1000
# ARCFACE_DISTANCES_FILE = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/input/MS-Celeb-1M/faces_emore/lfw_distances_arcface=1000class_acc=0.93833.npy'
# ARCFACE_DISTANCES_FILE = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/input/MS-Celeb-1M/faces_emore/calfw_distances_arcface=1000class_acc=0.82333.npy'
# ARCFACE_DISTANCES_FILE = '/home/bjgbiesseck/GitHub/InsightFace-tensorflow/output/dataset=WebFace260M_1000subj_classes=1000_backbone=resnet_v2_m_50_epoch-num=100_loss=arcface_s=64.0_m=0.5_moment=0.9_batch=64_lr-init=0.01_20230524-142404/lfw_distances_arcface=1000class_acc=0.90117.npy'

# NUM_CLASSES = 2000
# ARCFACE_DISTANCES_FILE = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/input/MS-Celeb-1M/faces_emore/lfw_distances_arcface=2000class_acc=0.96333.npy'
# ARCFACE_DISTANCES_FILE = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/input/MS-Celeb-1M/faces_emore/calfw_distances_arcface=2000class_acc=0.86750.npy'
# ARCFACE_DISTANCES_FILE = '/home/bjgbiesseck/GitHub/InsightFace-tensorflow/output/dataset=WebFace260M_2000subj_classes=2000_backbone=resnet_v2_m_50_epoch-num=100_loss=arcface_s=64.0_m=0.5_moment=0.9_batch=64_lr-init=0.01_20230524-190517/lfw_distances_arcface=2000class_acc=0.94350.npy'

NUM_CLASSES = 5000
# ARCFACE_DISTANCES_FILE = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/input/MS-Celeb-1M/faces_emore/lfw_distances_arcface=5000class_acc=0.97550.npy'
# ARCFACE_DISTANCES_FILE = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/input/MS-Celeb-1M/faces_emore/calfw_distances_arcface=5000class_acc=0.87900.npy'
ARCFACE_DISTANCES_FILE = '/home/bjgbiesseck/GitHub/InsightFace-tensorflow/output/dataset=WebFace260M_5000subj_classes=5000_backbone=resnet_v2_m_50_epoch-num=150_loss=arcface_s=64.0_m=0.5_moment=0.9_batch=64_lr-init=0.01_20230525-093855/lfw_distances_arcface=5000class_acc=0.96467.npy'

# NUM_CLASSES = 10000
# ARCFACE_DISTANCES_FILE = '/home/bjgbiesseck/datasets/MS-Celeb-1M/ms1m-retinaface-t1/lfw_distances_arcface=10000class_acc=0.98583.npy'



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


if FLAGS.dataset.upper() == 'reconst_mica_lfw'.upper():
    # DATA_PATH = os.path.join(ROOT_DIR, '../../BOVIFOCR_MICA_3Dreconstruction/demo/output/lfw')
    DATA_PATH = os.path.join(ROOT_DIR, '/nobackup/unico/datasets/face_recognition/MICA_3Dreconstruction/lfw')
    EVAL_DATASET = lfw_evaluation_3Dreconstructed_MICA_dataset_pairs.LFW_Evaluation_3D_Reconstructed_MICA_Dataset_Pairs(root=DATA_PATH, npoints=NUM_POINT, normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)
    
elif FLAGS.dataset.upper() == 'reconst_mica_calfw'.upper():
    DATA_PATH = os.path.join(ROOT_DIR, '../../BOVIFOCR_MICA_3Dreconstruction/demo/output/calfw')
    EVAL_DATASET = calfw_evaluation_3Dreconstructed_MICA_dataset_pairs.CALFW_Evaluation_3D_Reconstructed_MICA_Dataset_Pairs(root=DATA_PATH, npoints=NUM_POINT, normal_channel=FLAGS.normal, batch_size=BATCH_SIZE)



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

# Bernardo
def compute_all_embeddings_and_distances_pointnet2(sess, ops):
    is_training = False

    cur_batch_data = np.zeros((2,BATCH_SIZE,NUM_POINT,EVAL_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    all_distances = np.zeros((0), dtype=np.float32)
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
        print('all_distances.shape:', all_distances.shape, end='\r')
    print()

    EVAL_DATASET.reset()
    return all_distances


def fuse_scores(distances1, distances2):
    distances1 /= np.max(distances1)
    distances2 /= np.max(distances2)
    final_distances = (distances1 + distances2) / 2
    return final_distances


# Bernardo
def evaluate_varying_margin(num_votes):
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
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'embds': embds,
           'pred': pred,
           'loss': total_loss,
           'end_points': end_points}


    print('PointNet++ (3D) - Computing all embeddings and distances...')
    all_distances_pointnet2 = compute_all_embeddings_and_distances_pointnet2(sess, ops)

    print('ArcFace (2D) - Loading distances from file: ')
    all_distances_arcface = np.load(ARCFACE_DISTANCES_FILE)

    print('Fusing distances...')
    final_distances = fuse_scores(all_distances_arcface, all_distances_pointnet2)
    print('final_distances.shape:', final_distances.shape)

    min_margin, max_margin, step_margin = 0, 1, 0.005
    # min_margin, max_margin, step_margin = 0, 1, 0.01
    # min_margin, max_margin, step_margin = 0, 1, 0.05

    all_margins_eval = np.arange(min_margin, max_margin+step_margin, step_margin, dtype=np.float32)
    
    all_tp_eval = np.zeros_like(all_margins_eval)
    all_fp_eval = np.zeros_like(all_margins_eval)
    all_tn_eval = np.zeros_like(all_margins_eval)
    all_fn_eval = np.zeros_like(all_margins_eval)
    all_acc_eval = np.zeros_like(all_margins_eval)
    all_far_eval = np.zeros_like(all_margins_eval)
    all_tar_eval = np.zeros_like(all_margins_eval)

    for i, margin in enumerate(all_margins_eval):
        print(str(i) + '/' + str(len(all_margins_eval)-1) + ' - Evaluating dataset \'' + FLAGS.dataset + '\', margin=' + str(margin) + ' ...')
        tp, tn, fp, fn, acc, far, tar = eval_one_epoch(final_distances, margin)
        all_tp_eval[i] = tp
        all_tn_eval[i] = tn
        all_fp_eval[i] = fp
        all_fn_eval[i] = fn
        all_acc_eval[i] = acc
        all_far_eval[i] = far
        all_tar_eval[i] = tar
        print('    margin: %1.5f    tp: %d    tn: %d    fp: %d    fn: %d    acc: %1.5f    far: %1.5f    tar: %1.5f' % (margin, tp, tn, fp, fn, acc, far, tar))
        print('-------------------------')
    
    # print('Evaluation of dataset \'' + FLAGS.dataset + '\'')
    # for i, margin in enumerate(all_margins_eval):
    #     print('margin:', margin, '    tp:', all_tp_eval[i], '   tn', all_tn_eval[i], '   fp', all_fp_eval[i], '   fn', all_fn_eval[i], '   acc', all_acc_eval[i], '   far', all_far_eval[i], '   tar', all_tar_eval[i])

    path_file = '/'.join(MODEL_PATH.split('/')[:-1]) + '/' + 'evaluation_fusion_2D+3D_on_dataset=' + FLAGS.dataset + '.csv'
    print('Saving to CSV file:', path_file)
    save_metric_to_text_file(path_file, all_margins_eval, all_tp_eval, all_fp_eval, all_tn_eval, all_fn_eval, all_acc_eval, all_far_eval, all_tar_eval)


def cosine_distance(embds0, embds1):
    cos_dist = np.zeros(embds0.shape[0], dtype=np.float32)
    for i in range(cos_dist.shape[0]):
        cos_dist[i] = 1 - np.dot(embds0[i], embds1[i])/(np.linalg.norm(embds0[i])*np.linalg.norm(embds1[i]))
    return cos_dist

def eval_one_epoch(distances, margin=0.5):
    total_tp, total_tn, total_fp, total_fn, total_acc = 0, 0, 0, 0, 0
    total_far, total_tar = 0, 0

    total_correct = 0
    total_seen = 0
    # batch_idx = 0

    idx_dist = 0
    while EVAL_DATASET.has_next_batch():
        batch_data, batch_label = EVAL_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[1]    # Bernardo
        
        pred_labels = np.array([1 if d <= margin else 0 for d in distances[idx_dist:idx_dist+bsize]], dtype=np.int32)
        batch_label = np.array(batch_label[0:bsize], dtype=np.int32)

        correct = np.sum(pred_labels[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        # batch_idx += 1
        
        batch_tp = np.sum((pred_labels[0:bsize] == batch_label[0:bsize]) * (pred_labels[0:bsize] == 1))
        batch_tn = np.sum((pred_labels[0:bsize] == batch_label[0:bsize]) * (pred_labels[0:bsize] == 0))
        batch_fp = np.sum((pred_labels[0:bsize] != batch_label[0:bsize]) * (pred_labels[0:bsize] == 1))
        batch_fn = np.sum((pred_labels[0:bsize] != batch_label[0:bsize]) * (pred_labels[0:bsize] == 0))
        
        total_tp += batch_tp
        total_tn += batch_tn
        total_fp += batch_fp
        total_fn += batch_fn

        idx_dist += bsize

    total_acc = (float(total_tp) + float(total_tn)) / (float(total_tp) + float(total_tn) + float(total_fp) + float(total_fn))
    total_far = float(total_fp) / (float(total_fp) + float(total_tn))
    total_tar = float(total_tp) / (float(total_tp) + float(total_fn))
    
    EVAL_DATASET.reset()
    return total_tp, total_tn, total_fp, total_fn, total_acc, total_far, total_tar
    
    


if __name__=='__main__':
    with tf.Graph().as_default():
        # evaluate(num_votes=FLAGS.num_votes)
        evaluate_varying_margin(num_votes=FLAGS.num_votes)
    LOG_FOUT.close()
