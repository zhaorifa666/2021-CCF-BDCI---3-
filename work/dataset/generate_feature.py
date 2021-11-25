import numpy as np
import os
import argparse
from imblearn.over_sampling import RandomOverSampler

pairs_local = {'fsd': 
                ((0, 15, 16), (1, 2, 5), (2, 1, 3), (3, 2, 4), (4, 3, 3), (5, 1, 6), (6, 5, 7), (7, 6, 6), (8, 9, 12), (9, 8, 10), (10, 9, 11), (11, 10, 22),
                 (12, 8, 13), (13, 12, 14), (14, 13, 19), (15, 0, 17), (16, 0, 18), (17, 15, 15), (18, 16, 16), (19, 14, 20), (20, 19, 19), (21, 14, 14), 
                 (22, 11, 23), (23, 22, 22), (24, 11, 11))
        }
pairs_center1 = {'fsd': 
                ((0, 1, 8), (1, 8, 8), (2, 1, 8), (3, 1, 8), (4, 1, 8), (5, 1, 8), (6, 1, 8), (7, 1, 8), (8, 1, 1), (9, 1, 8), (10, 1, 8), (11, 1, 8),
                 (12, 1, 8), (13, 1, 8), (14, 1, 8), (15, 1, 8), (16, 1, 8), (17, 1, 8), (18, 1, 8), (19, 1, 8), (20, 1, 8), (21, 1, 8), 
                 (22, 1, 8), (23, 1, 8), (24, 1, 8))
        }
pairs_center2 = {'fsd': 
                ((8, 1, 0), (8, 1, 1), (8, 1, 2), (8, 1, 3), (8, 1, 4), (8, 1, 5), (8, 1, 6), (8, 1, 7), (1, 8, 8), (8, 1, 9), (8, 1, 10), (8, 1, 11),
                 (8, 1, 12), (8, 1, 13), (8, 1, 14), (8, 1, 15), (8, 1, 16), (8, 1, 17), (8, 1, 18), (8, 1, 19), (8, 1, 20), (8, 1, 21), 
                 (8, 1, 22), (8, 1, 23), (8, 1, 24))
        }
pairs_hands = {'fsd': 
                ((0, 4, 7), (1, 4, 7), (2, 4, 7), (3, 4, 7), (4, 7, 7), (5, 4, 7), (6, 4, 7), (7, 4, 4), (8, 4, 7), (9, 4, 7), (10, 4, 7), (11, 4, 7),
                 (12, 4, 7), (13, 4, 7), (14, 4, 7), (15, 4, 7), (16, 4, 7), (17, 4, 7), (18, 4, 7), (19, 4, 7), (20, 4, 7), (21, 4, 7), 
                 (22, 4, 7), (23, 4, 7), (24, 4, 7))
        }
pairs_elbows = {'fsd': 
                ((0, 3, 6), (1, 3, 6), (2, 3, 6), (3, 6, 6), (4, 3, 6), (5, 3, 6), (6, 3, 3), (7, 3, 6), (8, 3, 6), (9, 3, 6), (10, 3, 6), (11, 3, 6),
                 (12, 3, 6), (13, 3, 6), (14, 3, 6), (15, 3, 6), (16, 3, 6), (17, 3, 6), (18, 3, 6), (19, 3, 6), (20, 3, 6), (21, 3, 6), 
                 (22, 3, 6), (23, 3, 6), (24, 3, 6))
        }
pairs_knees = {'fsd': 
                ((0, 10, 13), (1, 10, 13), (2, 10, 13), (3, 10, 13), (4, 10, 13), (5, 10, 13), (6, 10, 13), (7, 10, 13), (8, 10, 13), (9, 10, 13), (10, 13, 13), (11, 10, 13),
                 (12, 10, 13), (13, 10, 10), (14, 10, 13), (15, 10, 13), (16, 10, 13), (17, 10, 13), (18, 10, 13), (19, 10, 13), (20, 10, 13), (21, 10, 13), 
                 (22, 10, 13), (23, 10, 13), (24, 10, 13))
        }
pairs_feet = {'fsd': 
                ((0, 22, 19), (1, 22, 19), (2, 22, 19), (3, 22, 19), (4, 22, 19), (5, 22, 19), (6, 22, 19), (7, 22, 19), (8, 22, 19), (9, 22, 19), (10, 22, 19), (11, 22, 19),
                 (12, 22, 19), (13, 22, 19), (14, 22, 19), (15, 22, 19), (16, 22, 19), (17, 22, 19), (18, 22, 19), (19, 22, 22), (20, 22, 19), (21, 22, 19),
                 (22, 19, 19), (23, 22, 19), (24, 22, 19)) 
        }
pairs_bone = {'fsd':
                ((0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 1), (9, 8), (10, 9), (11, 10),
                 (12, 8), (13, 12), (14, 13), (15, 0), (16, 0), (17, 15), (18, 16), (19, 14), (20, 19), (21, 14), 
                 (22, 11), (23, 22), (24, 11))
        }
def upsampling(data, labels):
    N, C, T, V, M = data.shape
    data = data.reshape(N, -1)
    ros = RandomOverSampler(random_state=0)
    X_resampled, Y_resampled = ros.fit_resample(data, labels)
    X_resampled = X_resampled.reshape(X_resampled.shape[0], C, T, V, M)
    return X_resampled, Y_resampled

def get_single_angle(vector, data):
    data = data[:, 0:2, :, :, :]
    N, C, T, V, M = data.shape
    info = vector['fsd']
    fp_sp = np.zeros((N, 1, T, V, M))
    for idx, (target, neigh1, neigh2) in enumerate(info):
        vec1 = data[:, :, :, neigh1, :] - data[:, :, :, target, :]
        vec2 = data[:, :, :, neigh2, :] - data[:, :, :, target, :]
        inner_product = (vec1 * vec2).sum(1)
        mod1 = np.clip(np.linalg.norm(vec1, ord=2, axis=1), 1e-6, np.inf)
        mod2 = np.clip(np.linalg.norm(vec2, ord=2, axis=1), 1e-6, np.inf)
        mod = mod1 * mod2
        theta = 1 - inner_product / mod
        theta = theta.reshape(N, 1, T, M)
        theta = np.clip(theta, 0, 2)
        fp_sp[:, :, :, idx, :] = theta
    return fp_sp

def get_JA(J, labels, fold_name, mode):
    N, C, T, V, M = J.shape
    save_data_path = './{}/JA{}.npy'.format(mode, fold_name)
    if mode != 'test':
        save_label_path = './{}/fold{}_label.npy'.format(mode, fold_name[-1])
    l = [pairs_local, pairs_center1, pairs_center2, pairs_hands, pairs_elbows, pairs_knees, pairs_feet]
    res = np.zeros((N, len(l), T, V, M), dtype='float32')
    for i, pairs in enumerate(l):
        ans = get_single_angle(pairs, J)
        res[:, i, :, :, :] = ans.squeeze(1)
    JA = np.concatenate((J , res), axis=1).astype('float32')
    if mode == 'train':
        JA, labels = upsampling(JA, labels)
    np.save(save_data_path, JA)
    if mode != 'test':
        np.save(save_label_path, labels)
    print('=========>> JA finished!!! <<=========')
    return JA

def get_J(JA, fold_name, mode):
    save_data_path = './{}/J{}.npy'.format(mode, fold_name)
    np.save(save_data_path, JA[:, 0:2, :, :, :].astype('float32'))
    print('=========>> J finished!!! <<=========')

def get_BA(JA, fold_name, mode):
    save_data_path = './{}/BA{}.npy'.format(mode, fold_name)
    J = JA[:, 0:2, :, :, :]
    N, C, T, V, M = J.shape
    pairs = pairs_bone['fsd']
    B = np.zeros((N, C, T, V, M), dtype='float32')
    for v1, v2 in pairs:
        B[:, :, :, v1, :] = J[:, :, :, v1, :] - J[:, :, :, v2, :]
    np.save(save_data_path, np.concatenate((B, JA[:, 2:9, :, :, :]), axis=1).astype('float32'))
    print('=========>> BA finished!!! <<=========')

def get_B(JA, fold_name, mode):
    save_data_path = './{}/B{}.npy'.format(mode, fold_name)
    J = JA[:, 0:2, :, :, :]
    N, C, T, V, M = J.shape
    pairs = pairs_bone['fsd']
    B = np.zeros((N, C, T, V, M), dtype='float32')
    for v1, v2 in pairs:
        B[:, :, :, v1, :] = J[:, :, :, v1, :] - J[:, :, :, v2, :]
    np.save(save_data_path, B.astype('float32'))
    print('=========>> B finished!!! <<=========')
    return B

def get_JB(JA, B, fold_name, mode):
    save_data_path = './{}/JB{}.npy'.format(mode, fold_name)
    J = JA[:, 0:2, :, :, :]
    np.save(save_data_path,  np.concatenate((J, B), axis=1).astype('float32'))
    print('=========>> JB finished!!! <<=========')

def get_JMj(JA, fold_name, mode):
    save_data_path = './{}/JMj{}.npy'.format(mode, fold_name)
    J = JA[:, 0:2, :, :, :]
    N, C, T, V, M = J.shape
    Mj = np.zeros((N, 2, T, V, M), dtype='float32')
    for t in range(1, T):
        Mj[:, :, t, :, :] = J[:, :, t, :, :] - J[:, :, t - 1, :, :]
    np.save(save_data_path,  np.concatenate((J, Mj), axis=1).astype('float32'))
    print('=========>> JMj finished!!! <<=========')

def get_Mb(B, fold_name, mode):
    save_data_path = './{}/Mb{}.npy'.format(mode, fold_name)
    N, C, T, V, M = B.shape
    Mb = np.zeros((N, 2, T, V, M), dtype='float32')
    for t in range(1, T):
        Mb[:, :, t, :, :] = B[:, :, t, :, :] - B[:, :, t - 1, :, :]
    np.save(save_data_path,  Mb.astype('float32'))
    print('=========>> Mb finished!!! <<=========')

def gen_train_valid(mode):
    data_path = '/home/aistudio/data/data118104/train_data.npy'
    labels_path = '/home/aistudio/data/data118104/train_label.npy'
    data = np.load(data_path)[:, 0:2, :, :, :].astype('float32')
    labels = np.load(labels_path)
    for idx in range(5):
        idx_path = './index/fold{}_{}_idx.npy'.format(idx, mode)
        fold_idx = np.load(idx_path)
        data_fold = data[fold_idx]
        labels_fold = labels[fold_idx]
        fold_name = '_fold{}'.format(idx)
        JA = get_JA(data_fold, labels_fold, fold_name, mode)
        get_J(JA, fold_name, mode)
        get_BA(JA, fold_name, mode)
        B = get_B(JA, fold_name, mode)
        get_JB(JA, B, fold_name, mode)
        get_JMj(JA, fold_name, mode)
        get_Mb(B, fold_name, mode)

def gen_test(mode):
    data_path = '/home/aistudio/data/data118104/test_B_data.npy'
    data = np.load(data_path)[:, 0:2, :, :, :].astype('float32')
    JA = get_JA(data, '', '', mode)
    get_J(JA, '', mode)
    get_BA(JA, '', mode)
    B = get_B(JA, '', mode)
    get_JB(JA, B, '', mode)
    get_JMj(JA, '', mode)
    get_Mb(B, '', mode)

def parse_args():
    parser = argparse.ArgumentParser("Generate feature script")
    parser.add_argument('-m',
                        '--mode',
                        type=str,
                        default='test',
                        choices=['train', 'valid', 'test'],
                        help='mode for generating features')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    mode_type = args.mode
    os.makedirs('./{}'.format(mode_type), exist_ok=True)
    if mode_type == 'train' or mode_type == 'valid':
        gen_train_valid(mode_type)
    else:
        gen_test(mode_type)
    