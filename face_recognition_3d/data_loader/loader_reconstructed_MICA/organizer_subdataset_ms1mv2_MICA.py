from __future__ import print_function

import os
import numpy as np
import sys

from tree_ms1mv2_3Dreconstructed_MICA import TreeMS1MV2_3DReconstructedMICA



def make_symbolic_links_folders(src_path='', folders_names=[''], limit_folders=10, dst_path=''):
    assert len(folders_names) >= limit_folders
    assert os.path.exists(src_path)
    if not os.path.exists(dst_path):
        print('Making destination folder \'' + dst_path + '\' ...')
        os.makedirs(dst_path)

    print('Making symbolic links in \'' + dst_path + '\' ...')
    for i, folder_name in enumerate(folders_names[:limit_folders]):
        src_folder_path = src_path + '/' + folder_name
        dst_folder_path = dst_path + '/' + folder_name
        command = 'ln -s ' + src_folder_path + ' ' + dst_folder_path
        print('%d/%d - %s' % (i+1, limit_folders, dst_folder_path), end='\r')
        os.system(command)
    print()


'''
if __name__ == '__main__':
    src_path = '/experiments/BOVIFOCR_project/datasets/MS-Celeb-1M/ms1m-retinaface-t1/images'    # RGB images
    # src_path = '/experiments/BOVIFOCR_project/datasets/faces/3D_reconstruction_MICA/output/MS-Celeb-1M/ms1m-retinaface-t1/images'    # Pointclouds
    
    # path_target_symb_links = '/experiments/BOVIFOCR_project/datasets/MS-Celeb-1M/ms1m-retinaface-t1/images_1000subj'
    path_target_symb_links = '/experiments/BOVIFOCR_project/datasets/MS-Celeb-1M/ms1m-retinaface-t1/images_2000subj'
    # path_target_symb_links = '/experiments/BOVIFOCR_project/datasets/faces/3D_reconstruction_MICA/output/MS-Celeb-1M/ms1m-retinaface-t1/images_10subj'
    # path_target_symb_links = '/experiments/BOVIFOCR_project/datasets/faces/3D_reconstruction_MICA/output/MS-Celeb-1M/ms1m-retinaface-t1/images_1000subj'
    # path_target_symb_links = '/experiments/BOVIFOCR_project/datasets/faces/3D_reconstruction_MICA/output/MS-Celeb-1M/ms1m-retinaface-t1/images_2000subj'
    # path_target_symb_links = '/experiments/BOVIFOCR_project/datasets/faces/3D_reconstruction_MICA/output/MS-Celeb-1M/ms1m-retinaface-t1/images_5000subj'

    dir_level = 1    # for RGB images
    # dir_level = 2  # for pointclouds

    file_ext = '.png'
    # file_ext = 'mesh.ply'
    # file_ext = '_centralized-nosetip_with-normals_filter-radius=100.npy'

    min_samples, max_samples = 2, -1

    # num_subjects_symb_links = 10
    # num_subjects_symb_links = 1000
    num_subjects_symb_links = 2000
    # num_subjects_symb_links = 5000
    # num_subjects_symb_links = 10000


    print('Searching all files ending with \'' + file_ext + '\' in \'' + src_path + '\' ...')
    # all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples = TreeMS1MV2_3DReconstructedMICA().get_all_pointclouds_paths_count(src_path, dir_level, file_ext)
    subjects_with_pc_paths, unique_subjects_names, samples_per_subject = TreeMS1MV2_3DReconstructedMICA().load_filter_organize_pointclouds_paths(src_path, dir_level, file_ext, min_samples, max_samples)

    # for uniq_subject_name, samp_per_subject in zip(unique_subjects_names, samples_per_subject):
    #     print('uniq_subject_name:', uniq_subject_name, '    samp_per_subject:', samp_per_subject)
    # print('len(subjects_with_pc_paths):', len(subjects_with_pc_paths))
    print('len(unique_subjects_names):', len(unique_subjects_names))
    print('len(samples_per_subject):', len(samples_per_subject))

    print('Making symbolic links...')
    make_symbolic_links_folders(src_path, unique_subjects_names, num_subjects_symb_links, path_target_symb_links)
'''


if __name__ == '__main__':
    src_path = '/experiments/BOVIFOCR_project/datasets/faces/MS-Celeb-1M/ms1m-retinaface-t1/images'    # RGB images
    # src_path = '/experiments/BOVIFOCR_project/datasets/faces/3D_reconstruction_MICA/output/MS-Celeb-1M/ms1m-retinaface-t1/images'    # Pointclouds
    
    dir_level = 1    # for RGB images
    # dir_level = 2  # for pointclouds

    file_ext = '.png'
    # file_ext = 'mesh.ply'
    # file_ext = '_centralized-nosetip_with-normals_filter-radius=100.npy'

    min_samples, max_samples = 2, -1



    print('Searching all files ending with \'' + file_ext + '\' in \'' + src_path + '\' ...')
    # all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, indexes_samples = TreeMS1MV2_3DReconstructedMICA().get_all_pointclouds_paths_count(src_path, dir_level, file_ext)
    subjects_with_pc_paths, unique_subjects_names, samples_per_subject = TreeMS1MV2_3DReconstructedMICA().load_filter_organize_pointclouds_paths(src_path, dir_level, file_ext, min_samples, max_samples)

    # for uniq_subject_name, samp_per_subject in zip(unique_subjects_names, samples_per_subject):
    #     print('uniq_subject_name:', uniq_subject_name, '    samp_per_subject:', samp_per_subject)
    # print('len(subjects_with_pc_paths):', len(subjects_with_pc_paths))
    print('len(unique_subjects_names):', len(unique_subjects_names))
    print('len(samples_per_subject):', len(samples_per_subject))




    path_target_symb_links = '/experiments/BOVIFOCR_project/datasets/faces/MS-Celeb-1M/ms1m-retinaface-t1/images_1000subj'
    num_subjects_symb_links = 1000
    print('\nMaking %s symbolic links...' % (num_subjects_symb_links))
    make_symbolic_links_folders(src_path, unique_subjects_names, num_subjects_symb_links, path_target_symb_links)

    path_target_symb_links = '/experiments/BOVIFOCR_project/datasets/faces/MS-Celeb-1M/ms1m-retinaface-t1/images_2000subj'
    num_subjects_symb_links = 2000
    print('\nMaking %s symbolic links...' % (num_subjects_symb_links))
    make_symbolic_links_folders(src_path, unique_subjects_names, num_subjects_symb_links, path_target_symb_links)

    path_target_symb_links = '/experiments/BOVIFOCR_project/datasets/faces/MS-Celeb-1M/ms1m-retinaface-t1/images_5000subj'
    num_subjects_symb_links = 5000
    print('\nMaking %s symbolic links...' % (num_subjects_symb_links))
    make_symbolic_links_folders(src_path, unique_subjects_names, num_subjects_symb_links, path_target_symb_links)

    path_target_symb_links = '/experiments/BOVIFOCR_project/datasets/faces/MS-Celeb-1M/ms1m-retinaface-t1/images_10000subj'
    num_subjects_symb_links = 10000
    print('\nMaking %s symbolic links...' % (num_subjects_symb_links))
    make_symbolic_links_folders(src_path, unique_subjects_names, num_subjects_symb_links, path_target_symb_links)

