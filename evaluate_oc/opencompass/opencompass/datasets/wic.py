import json

from datasets import Dataset, load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class WiCDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):

        if 'data_files' in kwargs:
            kwargs['data_files'] = get_data_path(kwargs['data_files'],
                                                 local_mode=True)
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            if example['label'] == 'true':
                example['answer'] = 1
            else:
                example['answer'] = 0

            return example

        dataset = dataset.map(preprocess)
        return dataset


@LOAD_DATASET.register_module()
class WiCDatasetV2(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                line['label'] = {'true': 'A', 'false': 'B'}[line['label']]
                dataset.append(line)
        return Dataset.from_list(dataset)
