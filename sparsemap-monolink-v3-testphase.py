import fasttext
import numpy as np
import os
import pickle
import torch as th
from torch.nn.parameter import Parameter
from torch.optim import SGD

from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

from transformers import BertModel, BertTokenizer

import io
import parameters
from activeset import ActiveSet
from embedding import *
from dataset import AlignDataset
from emalgos import EmAlgo
from concept import Concept
import utils
import embedding


def get_pair_hashcode(f,e):
    return "%s|||%s"%(f,e)

def test_alignment(align_f, concept, fs_test, es_test, out_filename, threshld=0.5):
    def align_matrix_to_alignment(S,threshld=0):
        res = []
        for j in range(S.shape[0]):
            for i in range(S.shape[1]):
                if S[j,i] >= threshld:
                    res.append('%d-%d'%(j+1,i+1))
        return ' '.join(res)
    with open(out_filename,'w') as outf:
        for f,e in zip(fs_test, es_test):
            eta = concept.eta_U(f,e,parameters.distortion)
            S = align_f(eta)
            print(align_matrix_to_alignment(S,threshld), file=outf)

def test_alignment_bpe(align_f, concept, fs_test, es_test, out_filename, threshld, src_idx, trg_idx):
    temp_file = 'align_temp_bpe.moses'
    test_alignment(align_f, concept, fs_test, es_test, temp_file, threshld)
    print('align finished')
    utils.converse_BPE_to_word(temp_file, src_idx, trg_idx, out_filename)
    print('convert bpe to aligndata finished')

def evaluate(align_f, concept, fs_test, es_test, true_label_file, threshld=0.5, bpe=True, src_idx=None, trg_idx=None):
    prediction_file = 'test-evaluate_temp.moses'
    if bpe:
        if not src_idx or not trg_idx:
            print('needs idx data')
            return
        test_alignment_bpe(align_f, concept, fs_test, es_test, prediction_file, threshld, src_idx, trg_idx)
    else:
        test_alignment(align_f, concept, fs_test, es_test, prediction_file, threshld)
    return utils.compute_aer(prediction_file, true_label_file)

def load_fasttext_embedding_model(f_fasttext_model_file, e_fasttext_model_file):
    if os.path.exists(f_fasttext_model_file):
        print('exist embedding src file')
        #f_model = fasttext.load_model(f_fasttext_model_file)
    if os.path.exists(e_fasttext_model_file):
        print('exist embedding trg file')
        #e_model = fasttext.load_model(e_fasttext_model_file)

    embedding_model = embedding.FastTextBPEEmbedding(f_fasttext_model_file, e_fasttext_model_file, parameters.normalize_embedding)
    return embedding_model

def main():

    emalgo = EmAlgo()
    active_set = ActiveSet(30)


    train_dataset = AlignDataset(parameters.f_train_filename, parameters.e_train_filename)
    test_dataset = AlignDataset(parameters.f_test_filename, parameters.e_test_filename)
    
    print('loading embedding model...')
    embedding_model = load_fasttext_embedding_model(parameters.f_fasttext_model_file, parameters.e_fasttext_model_file)
    print('OK! embedding model loaded')

    embedding_weight_file = ''

    embedding_model.load_weight(embedding_weight_file)


    score_before_init = evaluate(active_set.align_sparsemap,
                                 embedding_model,
                                 test_dataset.src_sentences,
                                 test_dataset.trg_sentences,
                                 parameters.true_label_file,
                                 threshld=0.5,
                                 bpe=parameters.bpe,
                                 src_idx=parameters.f_idx_data,
                                 trg_idx=parameters.e_idx_data)
    print('wedding embedding score before init : ', score_before_init)



    out_filename = ''

    test_alignment_bpe(active_set.align_sparsemap, 
                        embedding_model, 
                        test_dataset.src_sentences,
                        test_dataset.trg_sentences,
                        out_filename,
                        threshld=0.5,
                        src_idx=parameters.f_idx_data,
                        trg_idx=parameters.e_idx_data)

if __name__ == '__main__':
    main()



