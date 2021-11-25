# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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
from paddlevideo.utils import get_logger
from ..loader.builder import build_dataloader, build_dataset
from ..metrics import build_metric
from ..modeling.builder import build_model
from paddlevideo.utils import load

import numpy as np
import os
import paddle.nn.functional as F

logger = get_logger("paddlevideo")


@paddle.no_grad()
def test_model(cfg, weights, parallel=True):
    """Test model entry

    Args:
        cfg (dict): configuration.
        weights (str): weights path to load.
        parallel (bool): Whether to do multi-cards testing. Default: True.

    """
    # 1. Construct model.
    if cfg.MODEL.backbone.get('pretrained'):
        cfg.MODEL.backbone.pretrained = ''  # disable pretrain model init
    model = build_model(cfg.MODEL)
    if parallel:
        model = paddle.DataParallel(model)

    # 2. Construct dataset and dataloader.
    cfg.DATASET.test.test_mode = True
    dataset = build_dataset((cfg.DATASET.test, cfg.PIPELINE.test))
    batch_size = cfg.DATASET.get("test_batch_size", 8)
    places = paddle.set_device('gpu')
    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    num_workers = cfg.DATASET.get('test_num_workers', num_workers)
    dataloader_setting = dict(batch_size=batch_size,
                              num_workers=num_workers,
                              places=places,
                              drop_last=False,
                              shuffle=False)

    data_loader = build_dataloader(dataset, **dataloader_setting)

    model.eval()

    state_dicts = load(weights)
    model.set_state_dict(state_dicts)

    # add params to metrics
    cfg.METRIC.data_size = len(dataset)
    cfg.METRIC.batch_size = batch_size

    Metric = build_metric(cfg.METRIC)
    ans = np.zeros((len(data_loader), 30))
    for batch_id, data in enumerate(data_loader):
        outputs = model(data, mode='test')
        ans[batch_id, :] = outputs
        Metric.update(batch_id, data, outputs)
    os.makedirs('logits', exist_ok=True)
    with open('logits/{}.npy'.format(cfg.model_name), 'wb') as f:
        np.save(f, ans)
    Metric.accumulate()
