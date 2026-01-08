import os
import json
import hashlib
 
basic_path = "/public/dataset/generate_from_dpsk_r1/"

files = [os.path.join(basic_path, file) for file in os.listdir(basic_path)]

num_text = 0
text_dict = dict()
for file  in files:
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        text = json.loads(line.strip())['text']
        text_hash = hashlib.md5(text[:140].encode('utf-8')).hexdigest()
        text_dict[text_hash] = text
    num_text += len(lines)
    print(f"{num_text=} {len(text_dict)=}")
    # break

