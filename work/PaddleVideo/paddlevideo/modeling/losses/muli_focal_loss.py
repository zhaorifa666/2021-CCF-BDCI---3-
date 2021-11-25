# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..registry import LOSSES
from .base import BaseWeightedLoss


@LOSSES.register()
class MuliFocalLoss(BaseWeightedLoss):
    """Cross Entropy Loss."""
    def _estimate_difficulty_level(self, logits, labels):
        labels_onehot = F.one_hot(labels, 30)
        pt = labels_onehot * F.softmax(logits)
        difficulty_level = paddle.pow(1 - pt, 2)
        return difficulty_level, labels_onehot

    def _forward(self, logits, labels, **kwargs):
        """Forward function.
        Args:
            score (paddle.Tensor): The class score.
            labels (paddle.Tensor): The ground truth labels.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.
        Returns:
            loss (paddle.Tensor): The returned CrossEntropy loss.
        """
        logits = logits.astype('float32')
        difficulty_level, labels_onehot = self._estimate_difficulty_level(logits, labels)
        logs = nn.LogSoftmax(axis=1)(logits)
        labels_smooth = F.label_smooth(labels_onehot, epsilon=0.1)
        labels_smooth = paddle.squeeze(labels_smooth, axis=1)
        loss = -paddle.sum(difficulty_level * logs *labels_smooth, axis=1)
        return loss.mean()
