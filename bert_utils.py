import torch
import random


def chunks(lst, n):
    """ Yield successive n-sized chunks from lst
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def random_sample(inp, n_samples):
    if isinstance(inp, list):
        out = random.sample(inp, n_samples)

    if isinstance(inp, torch.Tensor):
        perm = torch.randperm(inp.size(0))
        idx = perm[:n_samples]
        out = inp[idx]

    return out

def tokenize_sentences(sentences, tokenizer, add_special_chars=True):
    tokenized_sentences = []
    for sent in sentences:
        if add_special_chars:
            sent = '[CLS] ' + sent + ' [SEP]'
        tokenized_sentences.append(tokenizer.tokenize(sent))
    return tokenized_sentences

def sentences_to_batch(sentences, tokenizer, add_special_chars=True, add_pad=True):
    tokenized_sentence = tokenize_sentences(sentences, tokenizer, add_special_chars)
    ids_sentences = []
    for sent in tokenized_sentence:
        ids_sentences.append(torch.tensor(tokenizer.convert_tokens_to_ids(sent)))
    if add_pad:
        ids_sentences = torch.nn.utils.rnn.pad_sequence(ids_sentences, batch_first=True, padding_value=0)
        return ids_sentences
    else:
        return torch.stack(ids_sentences)

def create_dummy_segment(sentence_tensor):
    return torch.zeros_like(sentence_tensor).as_type(sentence_tensor).to(sentence_tensor.device)
