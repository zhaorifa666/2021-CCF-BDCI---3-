#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import math
import random
import paddle
import paddle.nn.functional as F
from ..registry import PIPELINES
"""pipeline ops for Activity Net.
"""


@PIPELINES.register()
class UniformSampleFrames(object):
    """Uniformly sample frames from the video.
    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment. To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic.
    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "window_size", "frame_interval" and "num_clips".
    Args:
        window_size (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default: 1.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        seed (int): The random seed used during test time. Default: 255.
    """

    def __init__(self, window_size, num_clips=1, test_mode=False, seed=255):

        self.window_size = window_size
        self.num_clips = num_clips
        self.test_mode = test_mode
        self.seed = seed

    def get_frame_num(self, data):
        C, T, V, M = data.shape
        for i in range(T - 1, -1, -1):
            tmp = np.sum(data[0:2, i, :, :])
            if tmp != 0:
                T = i + 1
                break
        return T

    def _get_train_clips(self, num_frames, window_size):
        """Uniformly sample indices for training clips.
        Args:
            num_frames (int): The number of frames.
            window_size (int): The length of the clip.
        """

        assert self.num_clips == 1
        if num_frames < window_size:
            start = np.random.randint(0, num_frames)
            inds = np.arange(start, start + window_size)
        elif window_size <= num_frames < 2 * window_size:
            basic = np.arange(window_size)
            inds = np.random.choice(
                window_size + 1, num_frames - window_size, replace=False)
            offset = np.zeros(window_size + 1, dtype=np.int64)
            offset[inds] = 1
            offset = np.cumsum(offset)
            inds = basic + offset[:-1]
        else:
            bids = np.array(
                [i * num_frames // window_size for i in range(window_size + 1)])
            bsize = np.diff(bids)
            bst = bids[:window_size]
            offset = np.random.randint(bsize)
            inds = bst + offset
        return inds

    def _get_test_clips(self, num_frames, window_size):
        """Uniformly sample indices for testing clips.
        Args:
            num_frames (int): The number of frames.
            window_size (int): The length of the clip.
        """

        np.random.seed(self.seed)
        if num_frames < window_size:
            # Then we use a simple strategy
            if num_frames < self.num_clips:
                start_inds = list(range(self.num_clips))
            else:
                start_inds = [
                    i * num_frames // self.num_clips
                    for i in range(self.num_clips)
                ]
            inds = np.concatenate(
                [np.arange(i, i + window_size) for i in start_inds])
        elif window_size <= num_frames < window_size * 2:
            all_inds = []
            for i in range(self.num_clips):
                basic = np.arange(window_size)
                inds = np.random.choice(
                    window_size + 1, num_frames - window_size, replace=False)
                offset = np.zeros(window_size + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
                all_inds.append(inds)
            inds = np.concatenate(all_inds)
        else:
            bids = np.array(
                [i * num_frames // window_size for i in range(window_size + 1)])
            bsize = np.diff(bids)
            bst = bids[:window_size]
            all_inds = []
            for i in range(self.num_clips):
                offset = np.random.randint(bsize)
                all_inds.append(bst + offset)
            inds = np.concatenate(all_inds)
        return inds

    def __call__(self, results):
        data = results['data']
        num_frames = self.get_frame_num(data)

        if self.test_mode:
            inds = self._get_test_clips(num_frames, self.window_size)
        else:
            inds = self._get_train_clips(num_frames, self.window_size)

        inds = np.mod(inds, num_frames)
        assert inds.shape[0] == self.window_size
        data_pad = data[:, inds, :, :]
        results['data'] = data_pad
        return results


@PIPELINES.register()
class SkeletonNorm_J(object):
    def __init__(self, axis=2, squeeze=False):
        self.axis = axis
        self.squeeze = squeeze

    def __call__(self, results):
        data = results['data']

        # Centralization
        data[0:2, :, :, :] = data[0:2, :, :, :] - data[0:2, :, 8:9, :]
        data = data[:self.axis, :, :, :]
        C, T, V, M = data.shape
        if self.squeeze:
            data = data.reshape((C, T, V))  # M = 1
        results['data'] = data.astype('float32')
        if 'label' in results:
            label = results['label']
            results['label'] = np.expand_dims(label, 0).astype('int64')
        return results


@PIPELINES.register()
class SkeletonNorm_JA(object):
    def __init__(self, axis=2, squeeze=False):
        self.axis = axis
        self.squeeze = squeeze

    def __call__(self, results):
        data = results['data']

        # Centralization
        data[0:2, :, :, :] = data[0:2, :, :, :] - data[0:2, :, 8:9, :]
        C, T, V, M = data.shape
        if self.squeeze:
            data = data.reshape((C, T, V))  # M = 1
        results['data'] = data.astype('float32')
        if 'label' in results:
            label = results['label']
            results['label'] = np.expand_dims(label, 0).astype('int64')
        return results

@PIPELINES.register()
class SkeletonNorm_Mb(object):
    def __init__(self, axis=2, squeeze=False):
        self.axis = axis
        self.squeeze = squeeze

    def __call__(self, results):
        data = results['data']

        C, T, V, M = data.shape
        if self.squeeze:
            data = data.reshape((C, T, V))  # M = 1
        results['data'] = data.astype('float32')
        if 'label' in results:
            label = results['label']
            results['label'] = np.expand_dims(label, 0).astype('int64')
        return results


@PIPELINES.register()
class SkeletonNorm_JMj(object):
    def __init__(self, axis=2, squeeze=False):
        self.axis = axis
        self.squeeze = squeeze

    def __call__(self, results):
        data = results['data']

        # Centralization
        data[0:2, :, :, :] = data[0:2, :, :, :] - data[0:2, :, 8:9, :]
        C, T, V, M = data.shape
        if self.squeeze:
            data = data.reshape((C, T, V))  # M = 1
        results['data'] = data.astype('float32')
        if 'label' in results:
            label = results['label']
            results['label'] = np.expand_dims(label, 0).astype('int64')
        return results

@PIPELINES.register()
class SkeletonNorm_BA(object):
    """
    Normalize skeleton feature.
    Args:
        aixs: dimensions of vertex coordinate. 2 for (x,y), 3 for (x,y,z). Default: 2.
    """
    def __init__(self, axis=2, squeeze=False):
        self.axis = axis
        self.squeeze = squeeze

    def __call__(self, results):
        data = results['data']

        C, T, V, M = data.shape
        if self.squeeze:
            data = data.reshape((C, T, V))  # M = 1
        results['data'] = data.astype('float32')
        if 'label' in results:
            label = results['label']
            results['label'] = np.expand_dims(label, 0).astype('int64')
        return results

@PIPELINES.register()
class SkeletonNorm_B(object):
    def __init__(self, axis=2, squeeze=False):
        self.axis = axis
        self.squeeze = squeeze

    def __call__(self, results):
        data = results['data']

        # Centralization
        data = data[:self.axis, :, :, :]
        C, T, V, M = data.shape
        if self.squeeze:
            data = data.reshape((C, T, V))  # M = 1
        results['data'] = data.astype('float32')
        if 'label' in results:
            label = results['label']
            results['label'] = np.expand_dims(label, 0).astype('int64')
        return results

@PIPELINES.register()
class SkeletonNorm_JB(object):
    def __init__(self, axis=2, squeeze=False):
        self.axis = axis
        self.squeeze = squeeze

    def __call__(self, results):
        data = results['data']

        # Centralization
        data[0:2, :, :, :] = data[0:2, :, :, :] - data[0:2, :, 8:9, :]
        C, T, V, M = data.shape
        if self.squeeze:
            data = data.reshape((C, T, V))  # M = 1
        results['data'] = data.astype('float32')
        if 'label' in results:
            label = results['label']
            results['label'] = np.expand_dims(label, 0).astype('int64')
        return results


@PIPELINES.register()
class Iden(object):
    """
    Wrapper Pipeline
    """
    def __init__(self, label_expand=True):
        self.label_expand = label_expand

    def __call__(self, results):
        data = results['data']
        results['data'] = data.astype('float32')

        if 'label' in results and self.label_expand:
            label = results['label']
            results['label'] = np.expand_dims(label, 0).astype('int64')
        return results
