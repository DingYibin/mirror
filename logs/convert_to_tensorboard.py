import re
import os
from torch.utils.tensorboard import SummaryWriter

basic_path = "/workspace-dyb/mirror/logs/logs"


logpath = "/workspace-dyb/mirror/log_2025-1206-1238-40_1_2.log"
output_dir = "/workspace-dyb/mirror/logs/logs/eh_proj_add_embed"
output_dir_name = "eh_proj_add_embed"

logpath = "/workspace-dyb/mirror/log_2025-1205-2306-59_1_2.log"
output_dir = "/workspace-dyb/mirror/logs/logs/eh_proj_move"
output_dir_name = "eh_proj_move"

logpath = "/workspace-dyb/mirror/log_2025-1204-1541-09_1_2.log"
output_dir = "/workspace-dyb/mirror/logs/logs/eh_proj_move-old-lr"
output_dir_name = "eh_proj_move-old-lr"

logpath = "/workspace-dyb/mirror/log_2025-1203-1843-44_1_2.log"
output_dir = "/workspace-dyb/mirror/logs/logs/mtp-old-lr"
output_dir_name = "mtp-old-lr"

logpath = "/workspace-dyb/mirror/log_2025-1207-1254-42_1_2.log"
output_dir = "/workspace-dyb/mirror/logs/logs/mtp-remove-eh_proj"
output_dir_name = "mtp-remove-eh_proj"

output_dir = os.path.join(basic_path, output_dir_name)

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




