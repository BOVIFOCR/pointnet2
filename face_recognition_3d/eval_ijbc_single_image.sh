#!/bin/bash

# RESNET100 - Train: MS1MV3 - 1000 classes
export CUDA_VISIBLE_DEVICES=0; python eval_ijbc_single_image.py --num_class 1000 --model_path /home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/logs_training/classification/dataset=reconst_mica_ms1mv2_1000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=32_m=0.0_06052023_114705/model_best_train_accuracy.ckpt

# RESNET100 - Train: MS1MV3 - 2000 classes
export CUDA_VISIBLE_DEVICES=0; python eval_ijbc_single_image.py --num_class 2000 --model_path /home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/logs_training/classification/dataset=reconst_mica_ms1mv2_2000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=32_m=0.0_09062023_184940/model_best_train_accuracy.ckpt

# RESNET100 - Train: MS1MV3 - 5000 classes
export CUDA_VISIBLE_DEVICES=0; python eval_ijbc_single_image.py --num_class 5000 --model_path /home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/logs_training/classification/dataset=reconst_mica_ms1mv2_5000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-06_moment=0.9_loss=arcface_s=32_m=0.0_12062023_154451/model_best_train_accuracy.ckpt

# RESNET100 - Train: MS1MV3 - 10000 classes
export CUDA_VISIBLE_DEVICES=0; python eval_ijbc_single_image.py --num_class 10000 --model_path /home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/logs_training/classification/dataset=reconst_mica_ms1mv2_10000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=32_m=0.0_26052023_222414/model_best_train_accuracy.ckpt


#####################################


# RESNET100 - Train: WebFace260M - 1000 classes
export CUDA_VISIBLE_DEVICES=0; python eval_ijbc_single_image.py --num_class 1000 --model_path /home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/logs_training/classification/dataset=reconst_mica_webface_1000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=16_m=0.0_05062023_194932/model_best_train_accuracy.ckpt

# RESNET100 - Train: WebFace260M - 2000 classes
export CUDA_VISIBLE_DEVICES=0; python eval_ijbc_single_image.py --num_class 2000 --model_path /home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/logs_training/classification/dataset=reconst_mica_webface_2000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=16_m=0.0_05062023_213735/model_best_train_accuracy.ckpt

# RESNET100 - Train: WebFace260M - 5000 classes
export CUDA_VISIBLE_DEVICES=0; python eval_ijbc_single_image.py --num_class 5000 --model_path /home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/logs_training/classification/dataset=reconst_mica_webface_5000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-05_moment=0.9_loss=arcface_s=32_m=0.0_06062023_235151/model_best_train_accuracy.ckpt

# RESNET100 - Train: WebFace260M - 10000 classes
export CUDA_VISIBLE_DEVICES=0; python eval_ijbc_single_image.py --num_class 10000 --model_path /home/bjgbiesseck/GitHub/BOVIFOCR_pointnet2_tensorflow/face_recognition_3d/logs_training/classification/dataset=reconst_mica_webface_10000subj_model=pointnet2_cls_ssg_angmargin_max_epoch=100_lr-init=5e-06_moment=0.9_loss=arcface_s=32_m=0.0_13062023_123431/model_best_train_accuracy.ckpt

