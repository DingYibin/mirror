import re
import os
from torch.utils.tensorboard import SummaryWriter

basic_dir_path = "/workspace-dyb/mirror/logs/logs/tp8-dp2"

logpath = "/workspace-dyb/mirror/log_2025-1206-1238-40_1_2.log"
output_dir_name = "eh_proj_add_embed"

logpath = "/workspace-dyb/mirror/log_2025-1205-2306-59_1_2.log"
output_dir_name = "eh_proj_move"

logpath = "/workspace-dyb/mirror/log_2025-1204-1541-09_1_2.log"
output_dir_name = "eh_proj_move-old-lr"

logpath = "/workspace-dyb/mirror/log_2025-1203-1843-44_1_2.log"
output_dir_name = "mtp-old-lr"

logpath = "/workspace-dyb/mirror/log_2025-1207-1254-42_1_2.log"
output_dir_name = "mtp-remove-eh_proj"

logpath = "/workspace-dyb/mirror/log_2025-1208-1150-13_1_2.log"
output_dir_name = "eh_proj_add_hs"

basic_dir_path = "/workspace-dyb/mirror/logs/logs/tp4-dp4"

logpath = "/workspace-dyb/mirror/logs/logs-text/log_2025-1209-1745-21_1_2.log"
output_dir_name = "just_move_eh_proj"

logpath = "/workspace-dyb/mirror/logs/logs-text/log_2025-1209-2036-00_1_2.log"
output_dir_name = "mtp"

logpath = "/workspace-dyb/mirror/logs/logs-text/log_2025-1210-1910-19_1_2.log"
output_dir_name = "remove_embeddings"

logpath = "/workspace-dyb/mirror/logs/logs-text/log_2025-1212-0906-39_1_2.log"
output_dir_name = "remove_hiddenstates"


logpath = "/workspace-dyb/mirror/logs/logs-text/log_2025-1212-1250-25_1_2.log"
output_dir_name = "remove_self_attn"
output_dir = os.path.join(basic_dir_path, output_dir_name)

pattern = r'\[(.*?)\]\s+iteration\s+(\d+)/\s*\d+\s*\|\s*consumed samples:\s*(\d+)\s*\|\s*[^|]+\|\s*learning rate:\s*([\d\.E+-]+)\s*\|\s*[^|]+\|\s*lm loss:\s*([\d\.E+-]+)\s*\|\s*'


lines = []
with open(logpath, 'r', encoding='utf-8') as f:
    for line in f.readlines():
        line = line.strip()
        base_match = re.search(pattern, line)
        if base_match:
            lines.append(line)

print(len(lines))
def extract(pattern, line):
    match = re.search(pattern, line)
    return match.group(1) if match else None
iteration_pattern = r'iteration\s+(\d+)/\s*\d+'
# samples_pattern = r'consumed samples:\s*(\d+)'
# lr_pattern = r'learning rate:\s*([\d\.E+-]+)'
lm_loss_pattern = r'lm:\s*([\d\.E+-]+)'
# grad_norm_pattern = r'grad norm:\s*([\d\.]+)'
# loss_scale_pattern = r'loss scale:\s*([\d\.]+)'
record = [None] * len(lines)

loss_tags = ['lm'] + [f'mtp_{i}' for i in range(1, 8)]
loss_dict = {
    item : [None] * len(lines)
    for item in loss_tags
}
loss_dict['iteration'] = [None] * len(lines)
writers = {tag:SummaryWriter(os.path.join(output_dir, tag)) for tag in loss_tags}
for i, line in enumerate(lines):
    iteration = int(extract(iteration_pattern, line))
    for tag in loss_tags:
        loss_patern = tag + r' loss: ([\d\.E+-]+)'
        loss_value = extract(loss_patern, line)
        loss_value = float(loss_value) if loss_value is not None else 0.0
        writers[tag].add_scalar(f'loss', loss_value, iteration)

for tag in loss_tags:
    writers[tag].close()




