import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW
# from transformers import BertTokenizer
from utils import set_initial_random_seed, read_text_file_to_list, EarlyStopping, create_logger
from bert_utils import *
from metrics import AverageMeter, Accuracy

class COCOCaptionsDataset(Dataset):
    def __init__(self, tokenizer, file_path, phase, max_len=130, is_pad=False, padding_value=0):
        assert os.path.isfile(file_path)
        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if len(line) > 0]

        self.phase = phase
        self.max_len = min(len(max(lines, key=len)), max_len)
        self.is_pad = is_pad
        self.padding_value = padding_value

        self.examples = self._preprocess(lines, tokenizer)

    def _preprocess(self, lines, tokenizer):
        processed = []
        for line in lines:
            sent = '[CLS] ' + line + ' [SEP]'
            tokenized_line = tokenizer.tokenize(sent)
            ids_line = torch.tensor(tokenizer.convert_tokens_to_ids(tokenized_line))
            # Pad if needed
            if self.is_pad:
                if len(ids_line) < self.max_len:
                    diff = self.max_len - len(ids_line)
                    ids_line = torch.cat((ids_line, torch.empty((diff)).fill_(self.padding_value).type_as(ids_line)))

                elif len(ids_line) > self.max_len:
                    ids_line = ids_line[:self.max_len]

                assert len(ids_line) == self.max_len
            processed.append(ids_line)
        logging.info('Finished loading {} dataset'.format(self.phase))
        return processed

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

def collate(examples):
    return pad_sequence(examples, batch_first=True, padding_value=0)

def mask_tokens(inputs, tokenizer, params):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, params.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask).bool(), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1 # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def train_one_epoch(epoch, model, train_loader, optimizer, tokenizer, params):
    device = params.device
    avg_loss = AverageMeter()
    avg_acc = Accuracy(ignore_index=-1)

    model.train()
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        batch = batch.to(device)
        # segment = create_dummy_segment(batch)

        inputs, labels = mask_tokens(batch, tokenizer, params)
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs, masked_lm_labels=labels)
        loss, prediction_scores = outputs[:2]  # model outputs are always tuple in transformers (see doc)

        loss.backward()
        optimizer.step()

        avg_acc.update(prediction_scores.view(-1, params.vocab_size), labels.view(-1))
        avg_loss.update(loss.item())

    logging.info('Train-E-{}: loss: {:.4f}'.format(epoch, avg_loss()))

def evaluate(epoch, test_loader, tokenizer, params):
    device = params.device
    avg_loss = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):

            batch = batch.to(device)
            # segment = create_dummy_segment(batch)

            inputs, labels = mask_tokens(batch, tokenizer, params)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs, masked_lm_labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            avg_loss.update(loss.item())

    logging.info('Test-E-{}: loss: {:.4f}'.format(epoch, avg_loss()))
    return avg_loss()

def fit(params):
    logging.info('Start training for {} epochs:'.format(params.paramsn_epochs))

    if params.patience != None:
        early_stopping = EarlyStopping(params.patience)

    # if params.lr_scheduler:
    #     lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, params.step_size,
    #                                                    gamma=self.params.gamma)

    for epoch in range(params.n_epochs):
        logging.info('Epoch :: ' + str(epoch))

        train_one_epoch(epoch, model, train_dataloader, optimizer, bert_tokenizer, params)
        test_acc = evaluate(epoch, test_dataloader, bert_tokenizer, params)

        # if params.lr_scheduler:
        #     lr_scheduler.step()

    file_name = params.log_dir + '/epoch_{}_test_acc_{}_model.pth.tar'.format(epoch, str(round(test_acc, 4)))
    torch.save({'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optim_dict': optimizer.state_dict()}, file_name)

class Params():
    # data

    vocab_size = 30522
    mlm_probability = 0.15

    # optim
    learning_rate = 1e-3
    n_epochs = 30

    # exp
    root = Path(__file__).parent
    exp_name = 'test'
    log_dir = (root / 'exp_outputs' / exp_name).as_posix()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    set_initial_random_seed(42)
    params = Params()
    create_logger(params.log_dir)

    root = Path(__file__).parent
    train_filename = (root / 'data' / 'image_coco.txt').as_posix()
    test_filename = (root / 'data' / 'test_image_coco.txt').as_posix()

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = COCOCaptionsDataset(tokenizer=bert_tokenizer, file_path=train_filename, phase='train', is_pad=False)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=12, collate_fn=collate)

    test_dataset = COCOCaptionsDataset(tokenizer=bert_tokenizer, file_path=test_filename, phase='test', is_pad=False)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=12, collate_fn=collate)

    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    model = model.to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    for epoch in range(params.n_epochs):
        train_one_epoch(epoch, model, train_dataloader, optimizer, bert_tokenizer, params)
        evaluate(epoch, test_dataloader, bert_tokenizer, params)

