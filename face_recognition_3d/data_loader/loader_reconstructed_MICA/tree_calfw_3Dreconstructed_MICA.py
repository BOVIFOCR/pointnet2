from __future__ import print_function

import sys
import os
from glob import glob
from pathlib import Path

# print('os.path.dirname(os.path.abspath(__file__)):', os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from plots import plots_fr_pointnet2


# BERNARDO
class TreeCALFW_3DReconstructedMICA:
    
    def get_all_sub_folders(self, dir_path=''):
        return sorted(glob(dir_path + '/*/*/'))

    def get_sub_folders_one_level(self, dir_path=''):
        # sub_folders = [f.path for f in os.scandir(dir_path) if f.is_dir()]
        sub_folders = [f.name for f in os.scandir(dir_path) if f.is_dir()]
        return sorted(sub_folders)

    def get_all_pointclouds_paths(self, dir_path, pc_ext='.ply'):
        all_sub_folders = self.get_all_sub_folders(dir_path)
        all_pc_paths = []
        all_pc_subjects = []
        for sub_folder_pointcloud in all_sub_folders:
            pc_paths = sorted(glob(sub_folder_pointcloud + '/*' + pc_ext))
            # print('pc_paths:', pc_paths)
            pc_subjects = [pc_path.split('/')[-3] for pc_path in pc_paths]
            all_pc_paths += pc_paths
            all_pc_subjects += pc_subjects
        return all_pc_paths, all_pc_subjects

    def count_samples_per_subject(self, pc_paths_list=[''], pc_ext='.ply'):
        unique_subjects_names = []
        samples_per_subject = []
        for pc_path in sorted(pc_paths_list):
            data_path = pc_path.split('/')
            if pc_ext in data_path[-1]:
                subject_name = data_path[-3]
                if not subject_name in unique_subjects_names:
                    samples_per_subject.append(0)
                    unique_subjects_names.append(subject_name)
                samples_per_subject[-1] += 1
        assert len(unique_subjects_names) == len(samples_per_subject)
        return unique_subjects_names, samples_per_subject

    def get_all_pointclouds_paths_count(self, dir_path, pc_ext='.ply'):
        all_pc_paths, all_pc_subjects = self.get_all_pointclouds_paths(dir_path, pc_ext)
        unique_subjects_names, samples_per_subject = self.count_samples_per_subject(all_pc_paths, pc_ext)
        return all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject

    def filter_paths_by_minimum_samples(self, all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, pc_ext='.ply', min_samples=2, max_samples=-1):
        filtered_pc_paths = []
        filtered_pc_subjects = []
        filtered_subjects_names = []
        filtered_samples_per_subject = []
        selected_samples_per_subject = [0] * len(unique_subjects_names)
        for i, pc_path, pc_subj in zip(range(len(all_pc_paths)), all_pc_paths, all_pc_subjects):
            if samples_per_subject[unique_subjects_names.index(pc_subj)] >= min_samples and \
               (max_samples==-1 or selected_samples_per_subject[unique_subjects_names.index(pc_subj)] < max_samples):
                filtered_pc_paths.append(pc_path)
                filtered_pc_subjects.append(pc_subj)
                if not pc_subj in filtered_subjects_names:   # run once per subject
                    filtered_subjects_names.append(pc_subj)
                selected_samples_per_subject[unique_subjects_names.index(pc_subj)] += 1
        # filtered_samples_per_subject.append(samples_per_subject[unique_subjects_names.index(pc_subj)])
        filtered_samples_per_subject = [selected_samples_per_subject[unique_subjects_names.index(pc_subj)] for pc_subj in filtered_subjects_names]
        # print('selected_samples_per_subject:', selected_samples_per_subject)      
        return filtered_pc_paths, filtered_pc_subjects, filtered_subjects_names, filtered_samples_per_subject

    def load_filter_organize_pointclouds_paths(self, dir_path, pc_ext='.ply', min_samples=2, max_samples=-1):
        all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject = self.get_all_pointclouds_paths_count(dir_path, pc_ext)
        all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject = self.filter_paths_by_minimum_samples(all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, pc_ext, min_samples, max_samples)
        subjects_with_pc_paths = [()] * len(all_pc_paths)
        for i, pc_path, pc_subj in zip(range(len(all_pc_paths)), all_pc_paths, all_pc_subjects):
            subjects_with_pc_paths[i] = (pc_subj, pc_path)
        # print('samples_per_subject:', samples_per_subject)
        return subjects_with_pc_paths, unique_subjects_names, samples_per_subject

    def load_pairs_samples_protocol_from_file(self, protocol_file_path='pairs_CALFW.txt', dataset_path='', file_ext='.ply'):
        pos_pair_label = '1'
        neg_pair_label = '0'
        all_pos_pairs_paths = []
        all_neg_pairs_paths = []

        intermediate_dataset_path = 'images&landmarks/images&landmarks/images'

        with open(protocol_file_path, 'r') as fp:
            all_lines = [line.strip() for line in fp.readlines()]
            # all_lines = [line.rstrip('\r\n') for line in fp.readlines()]
            # print('all_lines:', all_lines)
            # sys.exit(0)

            for i in range(0,len(all_lines),2):
                sample_1, label1 = all_lines[i].split(' ')
                sample_2, label2 = all_lines[i+1].split(' ')
                assert label1 == label2

                sample_name1 = sample_1.split('.')[0]
                sample_name2 = sample_2.split('.')[0]
                # print('sample_name1:', sample_name1, '    label1: \''+label1+'\'')
                # print('sample_name2:', sample_name2, '    label2: \''+label2+'\'')

                path_sample1 = glob(os.path.join(dataset_path, intermediate_dataset_path, sample_name1, '*'+file_ext))[0]
                path_sample2 = glob(os.path.join(dataset_path, intermediate_dataset_path, sample_name2, '*'+file_ext))[0]

                if label1 != '0':   # positive pair
                    pos_pair = (pos_pair_label, path_sample1, path_sample2)
                    all_pos_pairs_paths.append(pos_pair)
                else:    # negative pair
                    neg_pair = (neg_pair_label, path_sample1, path_sample2)
                    all_neg_pairs_paths.append(neg_pair)

            return all_pos_pairs_paths, all_neg_pairs_paths, pos_pair_label, neg_pair_label


if __name__ == '__main__':
    dataset_path = '/home/bjgbiesseck/GitHub/MICA/demo/output/calfw'
    
    file_ext='.ply'
    
    min_samples=1
    # min_samples=3

    max_samples=-1
    # max_samples=100

    log_scale = True
    # log_scale = False

    # all_pc_paths, all_pc_subjects = TreeLFW_3DReconstructedMICA().get_all_pointclouds_paths(dir_path=dataset_path, pc_ext=file_ext)
    # unique_subjects_names, samples_per_subject = TreeLFW_3DReconstructedMICA().count_samples_per_subject(all_pc_paths, pc_ext=file_ext)
    # for pc_path, pc_subject in zip(all_pc_paths, all_pc_subjects):
    #     print('pc_path:', pc_path, '    pc_subject:', pc_subject)
    # for i, unique_subj_name in enumerate(unique_subjects_names):
    #     print('unique_subj_name', unique_subj_name, '    samples_per_subject:', samples_per_subject[i])
    # sys.exit(0)
    
    # print('Searching all files ending with \'' + file_ext + '\'...')
    # all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject = TreeLFW_3DReconstructedMICA().get_all_pointclouds_paths_count(dataset_path, file_ext)
    # # for pc_path, pc_subject in zip(all_pc_paths, all_pc_subjects):
    # #     print('pc_path:', pc_path, '    pc_subject:', pc_subject)
    # # sys.exit(0)
    # # for unique_subj_name, samples_per_subj in zip(unique_subjects_names, samples_per_subject):
    # #     print('unique_subj_name:', unique_subj_name, '    samples_per_subj:', samples_per_subj)
    # # sys.exit(0)

    # print('Searching all files ending with \'' + file_ext + '\'...')
    # all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject = TreeLFW_3DReconstructedMICA().filter_paths_by_minimum_samples(all_pc_paths, all_pc_subjects, unique_subjects_names, samples_per_subject, file_ext, min_samples)
    # # for pc_path, pc_subject in zip(all_pc_paths, all_pc_subjects):
    # #     print('pc_path:', pc_path, '    pc_subject:', pc_subject)
    # # sys.exit(0)
    # # for unique_subj_name, samples_per_subj in zip(unique_subjects_names, samples_per_subject):
    # #     print('unique_subj_name:', unique_subj_name, '    samples_per_subj:', samples_per_subj)
    # # print('len(unique_subj_name):', len(unique_subjects_names), '    len(samples_per_subj):', len(samples_per_subject))
    # # sys.exit(0)
    
    # print('Searching all files ending with \'' + file_ext + '\'...')
    # subjects_with_pc_paths, unique_subjects_names, samples_per_subject = TreeLFW_3DReconstructedMICA().load_filter_organize_pointclouds_paths(dataset_path, file_ext, min_samples, max_samples)
    # # for subj_pc_path in subjects_with_pc_paths:
    # #     print('subj_pc_path:', subj_pc_path)
    # # # sys.exit(0)

    # # unique_subjects_names = ['AAAAA', 'BBBB', 'CCCC', 'DDDD']
    # # samples_per_subject = [5, 2, 2, 1]
    # title = 'Dataset LFW - Samples per Subject'
    # subtitle = '(min_samples='+str(min_samples)+', max_samples='+str(max_samples)+')'
    # path_image = '/home/bjgbiesseck/GitHub/pointnet2_tf_original_biesseck/face_recognition_3d/logs_training/samples_per_subject_lfw_dataset_minsamples='+str(min_samples)+'_maxsamples='+str(max_samples)+'.png'
    # plots_fr_pointnet2.plot_samples_per_class_and_histogram(unique_subjects_names, samples_per_subject, log_scale, title=title, subtitle=subtitle, path_image=path_image, show_fig=False, save_fig=True)
    
    protocol_file_path = '/home/bjgbiesseck/GitHub/MICA/demo/input/calfw/pairs_CALFW.txt'
    all_pos_pairs_paths, all_neg_pairs_paths, pos_pair_label, neg_pair_label = TreeCALFW_3DReconstructedMICA().load_pairs_samples_protocol_from_file(protocol_file_path, dataset_path, file_ext)
    for pos_pairs in all_pos_pairs_paths:
        print('pos_pairs:', pos_pairs)
    # sys.exit(0)
    for neg_pairs in all_neg_pairs_paths:
        print('neg_pairs:', neg_pairs)
    sys.exit(0)