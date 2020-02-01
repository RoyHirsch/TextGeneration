from metrics.Metrics import Metrics

import torch
import random
import numpy as np
from scipy import linalg

from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
from bert_utils import *

def from_hidden_repr_to_sentence_embedding(hidden, method):
    """
        Generate embedding from the hidden representation generated from Bert model
        hidden.size() = [batch_size, max_len, embedding_size]
    """

    if method == 'mean':
        embeddings = hidden.mean(1)

    elif method == 'mean-no-special':
        hidden = hidden[:, 1:-1, :]
        embeddings = hidden.mean(1)

    elif method == 'cls':
        embeddings = hidden[:, 0, :]

    else:
        raise ValueError('Invalid aggregation method name: {}'.format(method))
    return embeddings

def get_bert_sentence_embeddings(bert_model, bert_tokenizer, sentences, method='mean'):
    """
        Process sentences. forward Bert model and extract sentence embedding
        :param bert_model: nn.Module - Bert
        :param bert_tokenizer:
        :param sentences: list of strings
        :param method: ['mean', 'mean-no-special', 'cls']
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Predict hidden states features for each layer
    bert_model.eval()
    with torch.no_grad():
        tokens_tensor = sentences_to_batch(sentences, bert_tokenizer)
        segments_tensors = create_dummy_segment(tokens_tensor)

        # See the models docstrings for the detail of the inputs
        if tokens_tensor.ndimension() == 1:
            tokens_tensor = tokens_tensor.unsqueeze(0)

        outputs = bert_model(tokens_tensor, token_type_ids=segments_tensors)
        # Get the last hidden state
        encoded_layers = outputs[0]

        embeddings = from_hidden_repr_to_sentence_embedding(encoded_layers, method)
    return embeddings

class BertDist(Metrics):
    def __init__(self, test_text, ref_text, batch_size, device, method='mean'):
        super().__init__()

        self.name = 'BertDist'
        self.ref_text = ref_text
        self.test_text = test_text
        self.batch_size = batch_size
        self.device = device
        self.method = method

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_embeddings(self, bert_model, sentences):
        sentences_chunks = chunks(sentences, self.batch_size)
        all_embeddings = []
        bert_model.eval()
        with torch.no_grad():
            for batch in sentences_chunks:
                tokens_tensor = sentences_to_batch(batch, bert_tokenizer)
                segments_tensors = create_dummy_segment(tokens_tensor)

                # See the models docstrings for the detail of the inputs
                if tokens_tensor.ndimension() == 1:
                    tokens_tensor = tokens_tensor.unsqueeze(0)

                outputs = bert_model(tokens_tensor, token_type_ids=segments_tensors)
                # Get the last hidden state
                encoded_layers = outputs[0]

                embeddings = from_hidden_repr_to_sentence_embedding(encoded_layers, self.method)
                all_embeddings.append(embeddings)
        return torch.cat(all_embeddings)

    def get_score(self, bert_model, limit=None):

        if limit:
            ref_text = random_sample(self.ref_text, limit)
            test_text = random_sample(self.test_text, limit)
        else:
            ref_text = self.ref_text
            test_text = self.test_text

        ref_embds = self.get_embeddings(bert_model, ref_text)
        test_embds = self.get_embeddings(bert_model, test_text)

        assert ref_embds.shape == test_embds.shape, "Training and test mean embedding tensors have different shapes"
        score = self.calculate_fid(ref_embds, test_embds)
        return score

    def calculate_fid(self, embds1, embds2, eps=1e-6):
        """
        Calculates the Frechet distance between two multivariate Gaussians:
            X_1 ~ N(mu_1, C_1)
            and X_2 ~ N(mu_2, C_2) as:
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        embds1.size() = embds2.size() = [n_samples, embedding_size]

        # https://github.com/hukkelas/pytorch-frechet-inception-distance/blob/master/fid.py
        """
        if not(isinstance(embds1, np.ndarray)):
            embds1 = embds1.detach().cpu().numpy()

        if not(isinstance(embds2, np.ndarray)):
            embds2 = embds2.detach().cpu().numpy()
        # calculate mean and covariance statistics
        mu1, sigma1 = embds1.mean(axis=0), np.cov(embds1, rowvar=False)
        mu2, sigma2 = embds2.mean(axis=0), np.cov(embds2, rowvar=False)

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

        diff = mu1 - mu2
        # product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            print("fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        score = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean
        return score


if __name__ == '__main__':
    from pathlib import Path
    from utils import set_initial_random_seed, read_text_file_to_list
    set_initial_random_seed(42)

    root = Path(__file__).parent.parent
    test_files = ['mle.txt', 'leakgan.txt', 'maligan.txt',
                  'rank.txt', 'seq.txt', 'textgan.txt']

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    scores = {}
    for file in test_files:
        name = file.split('.')[0]
        f_name = (root / 'outputs' / file).as_posix()
        mle_txt = read_text_file_to_list(f_name)
        f_name = (root / 'data' / 'test_image_coco.txt').as_posix()
        ref_txt = read_text_file_to_list(f_name)

        device = 'gpu' if torch.cuda.is_available() else 'cpu'

        dist = BertDist(test_text=mle_txt, ref_text=ref_txt, batch_size=10, device=device, method='mean')
        print('Start calc for ' + name)
        score = dist.get_score(model, None)
        scores[name] = score
        print('{} score: {:.4f}'.format(name, score))

    print(scores)
    # leakgan: 14.166315642521123 mle: 11.602150307720777 score_ref: 3.45060815012635
