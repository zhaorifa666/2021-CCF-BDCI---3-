import os
import re
import numpy as np
import csv


def softmax(X):
    m = np.max(X, axis=1, keepdims=True)
    exp_X = np.exp(X - m)
    exp_X = np.exp(X)
    prob = exp_X / np.sum(exp_X, axis=1, keepdims=True)
    return prob


def is_Mb(file_name):
    pattern = 'CTRGCN_Mb_fold\d+\.npy'
    return re.match(pattern, file_name) is not None


output_prob = None
folder = './logits'
for logits_file in os.listdir(folder):
    logits = np.load(os.path.join(folder, logits_file))
    prob = softmax(logits)
    if is_Mb(logits_file):
        prob *= 0.7
    if output_prob is None:
        output_prob = prob
    else:
        output_prob = output_prob + prob
pred = np.argmax(output_prob, axis=1)

with open('./submission_ensemble.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(('sample_index', 'predict_category'))
    for i, p in enumerate(pred):
        writer.writerow((i, p))
