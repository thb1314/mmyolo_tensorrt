# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import Mock

import torch
from mmengine.config import Config
from mmengine.runner import Runner
from torch.utils.data import Dataset

from mmyolo.engine.hooks import YOLOXModeSwitchHook
from mmyolo.utils import register_all_modules


class DummyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


pipeline1 = [
    dict(type='mmdet.Resize'),
]

pipeline2 = [
    dict(type='mmdet.RandomFlip'),
]
register_all_modules()


class TestYOLOXModeSwitchHook(TestCase):

    def test(self):
        train_dataloader = dict(
            dataset=DummyDataset(),
            sampler=dict(type='DefaultSampler', shuffle=True),
            batch_size=3,
            num_workers=0)

        runner = Mock()
        runner.model = Mock()
        runner.model.module = Mock()

        runner.model.bbox_head.use_bbox_aux = False
        runner.cfg.train_dataloader = Config(train_dataloader)
        runner.train_dataloader = Runner.build_dataloader(train_dataloader)
        runner.train_dataloader.dataset.pipeline = pipeline1

        hook = YOLOXModeSwitchHook(
            num_last_epochs=15, new_train_pipeline=pipeline2)

        # test after change mode
        runner.epoch = 284
        runner.max_epochs = 300
        hook.before_train_epoch(runner)
        self.assertTrue(runner.model.bbox_head.use_bbox_aux)
        self.assertEqual(runner.train_loop.dataloader.dataset.pipeline,
                         pipeline2)
