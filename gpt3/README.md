# Experiments
## Qwen3
Although the name of dir is Gpt3, only Qwen3 is supported now.


| Model Size | TP size | DP size | PP size | seq lens | global batch size | max memory per gpu | time per iteration(ms) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0.6B | 1 | 8 | 1 | 4096 | 256 |  << 140GB |
| 1.7B | 1 | 8 | 1 | 4096 | 256 | 88856MiB | 5000+ |
| 8B | 2 | 4 | 1 | 4096 | 256 | 134728MiB | 14400(+-) |


