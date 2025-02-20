from __future__ import print_function

import os
import numpy as np
import sys

from tree_ms1mv2_3Dreconstructed_MICA import TreeMS1MV2_3DReconstructedMICA
from tree_webface_3Dreconstructed_MICA import TreeWEBFACE_3DReconstructedMICA



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



if __name__ == '__main__':
    # # For RGB images
    # src_path = '/home/bjgbiesseck/datasets/MS-Celeb-1M/ms1m-retinaface-t1/images'    # RGB images
    # dir_level = 1    # for RGB images
    # file_ext = '.png'
    
    # For 3D pointclouds
    # src_path = '/home/bjgbiesseck/GitHub/BOVIFOCR_MICA_3Dreconstruction/demo/output/MS-Celeb-1M_3D_reconstruction_originalMICA/ms1m-retinaface-t1/images'   # duo
    src_path = '/datasets1/bjgbiesseck/WebFace260M/3D_reconstruction/MICA_original'    # duo
    # src_path = '/nobackup/unico/datasets/face_recognition/MICA_3Dreconstruction/WebFace260M_3D_reconstruction_originalMICA/MICA_original'    # diolkos
    # src_path = '/nobackup1/bjgbiesseck/datasets/MS-Celeb-1M_3D_reconstruction_originalMICA/images'             # peixoto
    # src_path = '/nobackup1/bjgbiesseck/datasets/WebFace260M_3D_reconstruction_originalMICA/MICA_original'      # peixoto
    dir_level = 2      # for pointclouds
    file_ext = '_centralized-nosetip_with-normals_filter-radius=100.npy'
    # file_ext = 'mesh.ply'

    min_samples, max_samples = 2, -1

    # nums_subjects_symb_links = [1000, 2000, 5000]
    nums_subjects_symb_links = [1000, 2000, 5000, 10000]



    print('Searching all files ending with \'' + file_ext + '\' in \'' + src_path + '\' ...')
    subjects_with_pc_paths, unique_subjects_names, samples_per_subject = TreeMS1MV2_3DReconstructedMICA().load_filter_organize_pointclouds_paths(src_path, dir_level, file_ext, min_samples, max_samples)

    print('len(unique_subjects_names):', len(unique_subjects_names))
    print('len(samples_per_subject):', len(samples_per_subject))

    for num_subjects in nums_subjects_symb_links:
        path_target_symb_links = '/'.join(src_path.split('/')[:-1]) + '/images_' + str(num_subjects) + 'subj'
        print('\nMaking %s symbolic links...' % (num_subjects))
        make_symbolic_links_folders(src_path, unique_subjects_names, num_subjects, path_target_symb_links)

