import numpy as np
from sklearn.model_selection import StratifiedKFold

X = np.load('/home/aistudio/data/data118104/train_data.npy')
y = np.load('/home/aistudio/data/data118104/train_label.npy')
skf = StratifiedKFold(n_splits=5, shuffle=True)
res = skf.split(X, y)
for idx, (train_index, valid_index) in enumerate(res):
    train_index_path = '/home/aistudio/work/dataset/index/fold{}_train_idx.npy'.format(idx)
    valid_index_path = '/home/aistudio/work/dataset/index/fold{}_valid_idx.npy'.format(idx)
    np.save(train_index_path, train_index)
    np.save(valid_index_path, valid_index)