import os
import glob
import numpy as np
import scipy.io as scio
import pickle


class Label:
    
    def __init__(self, config):
        root = config.DATASET.ROOT
        dataset_name = config.DATASET.DATASET
        if dataset_name == 'shanghai':
            self.frame_mask = os.path.join(root, dataset_name, 'test_frame_mask/*')
        elif dataset_name == 'ped2':
            jsonName = dataset_name + '.json'
            self.jsonPath = os.path.join(root, dataset_name, jsonName)
            self.METADATA = {
            "ped2": {
            "testing_video_num": 12,
            "testing_frames_cnt": [180, 180, 150, 180, 150, 180, 180, 180, 120, 150,
                                   180, 180]
                },
            }

        mat_name = dataset_name + '.mat'

        self.mat_path = os.path.join(root, dataset_name, mat_name)

        test_set = config.DATASET.TESTSET
        test_dataset_path = os.path.join(root, dataset_name, test_set)
        video_folders = (os.listdir(test_dataset_path))
        video_folders.sort()
        self.video_folders = [os.path.join(test_dataset_path, folder) for folder in video_folders]
        self.dataset_name = dataset_name

    def __call__(self):
        if self.dataset_name == 'shanghai':
            np_list = glob.glob(self.frame_mask)
            np_list.sort()

            gt = []
            for npy in np_list:
                gt.append(np.load(npy))

            return gt
        elif self.dataset_name == 'ped2':
            gt = pickle.load(open(self.jsonPath, "rb"))
            gt_concat = np.concatenate(list(gt.values()), axis=0)
            # print(gt_concat.shape)
            # print(type(gt_concat))

            new_gt = []
            start_idx = 0
            for cur_video_id in range(self.METADATA[self.dataset_name]["testing_video_num"]):
                gt_each_video = gt_concat[start_idx:start_idx + self.METADATA[self.dataset_name]["testing_frames_cnt"][cur_video_id]]
                start_idx += self.METADATA[self.dataset_name]["testing_frames_cnt"][cur_video_id]
                # print(gt_each_video)
                new_gt.append(gt_each_video.astype("int32").tolist())

            # print(type(new_gt))
            # print(len(new_gt))
            # print(new_gt[0])

            return new_gt

        else:
            abnormal_mat = scio.loadmat(self.mat_path, squeeze_me=True)['gt']

            all_gt = []
            
            
            for i in range(abnormal_mat.shape[0]):
                length = len(os.listdir(self.video_folders[i]))
                sub_video_gt = np.zeros((length,), dtype=np.int8)

                one_abnormal = abnormal_mat[i]
                if one_abnormal.ndim == 1:
                    one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))

                for j in range(one_abnormal.shape[1]):
                    start = one_abnormal[0, j] - 1   # TODO
                    end = one_abnormal[1, j]

                    sub_video_gt[start: end] = 1

                all_gt.append(sub_video_gt)

            return all_gt
