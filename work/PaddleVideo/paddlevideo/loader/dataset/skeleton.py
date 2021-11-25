# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import os.path as osp
import copy
import random
import numpy as np
import pickle

from ..registry import DATASETS
from .base import BaseDataset
from ...utils import get_logger

logger = get_logger("paddlevideo")

# #set random seed
# random.seed(0)
# np.random.seed(0)
# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import os.path as osp
import copy
import random
import numpy as np
import pickle
import gc

from ..registry import DATASETS
from .base import BaseDataset
from ...utils import get_logger

logger = get_logger("paddlevideo")


@DATASETS.register()
class KFoldSkeletonDataset(BaseDataset):
    """
    Skeleton dataset for action recognition.
    The dataset loads skeleton feature, and apply norm operatations.
    Args:
        file_path (str): Path to the index file.
        pipeline(obj): Define the pipeline of data preprocessing.
        data_prefix (str): directory path of the data. Default: None.
        test_mode (bool): Whether to bulid the test dataset. Default: False.
    """
    def __init__(self, index_path, pipeline, file_path='/home/zhaorifa/competition/home/aistudio/work/data/train_data_padd.npy',
                label_path='/home/zhaorifa/competition/home/aistudio/work/data/train_label.npy', test_mode=False):
        self.file_path = file_path
        if not test_mode:
            self.label_path = label_path
        self.index_path = index_path
        super().__init__(file_path, pipeline, test_mode=test_mode)

    def load_file(self):
        """Load feature file to get skeleton information."""
        logger.info("Loading data, it will take some moment...")
        self.idx = np.load(self.index_path)
        tmp_data = np.load(self.file_path)
        self.data = tmp_data[self.idx]
        del tmp_data
        gc.collect()
        if self.label_path:
            if self.label_path.endswith('npy'):
                self.label = np.load(self.label_path)
            elif self.label_path.endswith('pkl'):
                with open(self.label_path, 'rb') as f:
                    sample_name, self.label = pickle.load(f)
            self.label = self.label[self.idx]
        else:
            logger.info(
                "Label path not provided when test_mode={}, here just output predictions."
                .format(self.test_mode))
        logger.info("Data Loaded!")
        return self.data  # used for __len__

    def prepare_train(self, idx):
        """Prepare the feature for training/valid given index. """
        results = dict()
        results['data'] = copy.deepcopy(self.data[idx])
        results['label'] = copy.deepcopy(self.label[idx])
        results = self.pipeline(results)
        return results['data'], results['label']

    def prepare_test(self, idx):
        """Prepare the feature for test given index. """
        results = dict()
        results['data'] = copy.deepcopy(self.data[idx])
        if self.label_path:
            results['label'] = copy.deepcopy(self.label[idx])
            results = self.pipeline(results)
            return results['data'], results['label']
        else:
            results = self.pipeline(results)
            return [results['data']]


@DATASETS.register()
class SkeletonDataset(BaseDataset):
    """
    Skeleton dataset for action recognition.
    The dataset loads skeleton feature, and apply norm operatations.
    Args:
        file_path (str): Path to the index file.
        pipeline(obj): Define the pipeline of data preprocessing.
        data_prefix (str): directory path of the data. Default: None.
        test_mode (bool): Whether to bulid the test dataset. Default: False.
    """
    def __init__(self, file_path, pipeline, label_path=None, test_mode=False):
        self.label_path = label_path
        super().__init__(file_path, pipeline, test_mode=test_mode)

    def load_file(self):
        """Load feature file to get skeleton information."""
        logger.info("Loading data, it will take some moment...")
        self.data = np.load(self.file_path)
        if self.label_path:
            if self.label_path.endswith('npy'):
                self.label = np.load(self.label_path)
            elif self.label_path.endswith('pkl'):
                with open(self.label_path, 'rb') as f:
                    sample_name, self.label = pickle.load(f)
        else:
            logger.info(
                "Label path not provided when test_mode={}, here just output predictions."
                .format(self.test_mode))
        logger.info("Data Loaded!")
        return self.data  # used for __len__

    def prepare_train(self, idx):
        """Prepare the feature for training/valid given index. """
        results = dict()
        results['data'] = copy.deepcopy(self.data[idx])
        results['label'] = copy.deepcopy(self.label[idx])
        results = self.pipeline(results)
        return results['data'], results['label']

    def prepare_test(self, idx):
        """Prepare the feature for test given index. """
        results = dict()
        results['data'] = copy.deepcopy(self.data[idx])
        if self.label_path:
            results['label'] = copy.deepcopy(self.label[idx])
            results = self.pipeline(results)
            return results['data'], results['label']
        else:
            results = self.pipeline(results)
            return [results['data']]
