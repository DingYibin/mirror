

分别处理数据集
```bash
uv run python data/convert_distill_to_text.py
```
合并数据集

```bash
uv run python /private/mirror/thirdparty/Megatron-LM/tools/merge_datasets.py \
--input /private/converted_dataset/a-m-team/AM-DeepSeek-Distilled-40M \
--output-prefix /private/converted_dataset/a-m-team/merged-r1-dataset

```