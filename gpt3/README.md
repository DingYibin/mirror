# Experiments
## Qwen3
Although the name of dir is Gpt3, only Qwen3 is supported now.


| Model Size | TP size | seq lens | batch size per dp | max memory per gpu | time per iteration(ms) |
|-------|-------|-------|-------|-------|-------|
| 0.6B | 1 | 4096 | 32 |  << 140GB |
| 1.7B | 1 | 4096 | 32 | 88856MiB | 5000+ |
| 8B | 2 | 4096 | 64 | 134728MiB | 14400(+-) |


