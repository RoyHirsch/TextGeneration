import torch
from metrics import SelfBleu
import sys
import os
from pathlib import Path
sys.path.append(Path(__file__).parent.as_posix())

def read_text_file_to_list(f_name):
    with open(f_name, 'r') as f:
        data = f.readlines()
    return data

root = Path(__file__).parent
f_name = (root / 'outputs' / 'mle.txt').as_posix()
mle_txt = read_text_file_to_list(f_name)
self_bleu = SelfBleu(test_text=f_name)
self_bleu.get_bleu(limit=10)


