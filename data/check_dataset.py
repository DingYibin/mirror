from megatron.core.datasets.indexed_dataset import _IndexReader


data_index_file = "/workspace-dyb/converted_dataset/a-m-team/merged-r1-dataset.idx"

index = _IndexReader(data_index_file, False)