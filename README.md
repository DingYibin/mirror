
```bash
gpt3/train_gpt3_with_qwen3_tokenizer.sh \
    /private/experiments/qwq-tp8/ckpt \
    /private/experiments/qwq-tp8/log \
    /private/converted_dataset/shareAI/ShareGPT-Chinese-English-90k/sharegpt_jsonl/processed_data_text_document \
    QWQ32B \
    1 \
    1 \
    &> single.log & tail -f single.log
```

## notes
### 2025-11-29
- lr 最大 1e-5， 最小 1e-6 明显太小，cosine衰减500步，参考specforge的数据，1e-4，调整最大为1e-4, 最小1e-5,衰减3000步
- 上一设定看起来依旧太小，是lr的问题还是mtp层太多的问题？尝试改7层MTP为2层，重启实验
- 7改2无用，注意到有一个参数--mtp-loss-scaling-factor控制了loss的大小，也会影响grad，默认值为0.1，修改为1.0

以上都没用，因为loss写错了

###
- 标准7层MTP，训练日志：log_2025-1203-1843-44_1_2.log
- 移动 eh_proj 的MTP训练日志 log_2025-1204-1541-09_1_2.log
- log_2025-1205-2306-59_1_2 ： 移动 eh_proj，修改了学习率的schedule
    - 比较loss下降曲线，学习率的变小，导致loss下降变慢
- 倾向认为改学习率无效，修改eh_proj后加到embeddings上后再进行norm计算, log_2025-1206-1238-40_1_2
- 怀疑eh_proj不在前面时没啥用，remove eh_proj后尝试 log_2025-1207-1254-42_1_2
    - remove eh proj 未结束，参看log，move还是有用的
- eh_proj 加到 hiddenstates后再过norm， mode 3，训练， log_2025-1208-1150-13_1_2.log