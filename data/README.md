

分别处理数据集
```bash
uv run python data/convert_distill_to_text.py
```
合并数据集

```bash
uv run python /workspace-dyb/mirror/thirdparty/Megatron-LM/tools/merge_datasets.py \
--input /workspace-dyb/converted_dataset/a-m-team/AM-DeepSeek-Distilled-40M \
--output-prefix /workspace-dyb/converted_dataset/a-m-team/merged-r1-dataset


uv run python /workspace-dyb/mirror/thirdparty/Megatron-LM/tools/merge_datasets.py \
--input /workspace-dyb/converted_dataset/generate_from_dpsk_r1/split \
--output-prefix /workspace-dyb/converted_dataset/merged/random-distill


python /workspace-dyb/mirror/thirdparty/Megatron-LM/tools/merge_datasets.py \
--input /workspace-dyb/converted_dataset/merged \
--output-prefix /workspace-dyb/converted_dataset/all
```

数据集说明
1. /workspace-dyb/converted_dataset下均为基于qwen tokenizer转换得到的数据集
2. /workspace-dyb/converted_dataset下为各个数据集各自合并的megatron格式的数据集
3. /workspace-dyb/converted_dataset/all为所有数据集合并的数据集