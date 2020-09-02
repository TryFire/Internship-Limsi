import fasttext
import numpy as np
import os

import torch as th
from torch.nn.parameter import Parameter
from torch.optim import SGD

from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment

from transformers import BertModel, BertTokenizer

import io

from activeset import ActiveSet
from embedding import *
from dataset import AlignDataset
from emalgos import EmAlgo
from concept import Concept
import utils 
import embedding
def get_pair_hashcode(f,e):
    return "%s|||%s"%(f,e)

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data

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
            eta = concept.eta_U(f,e)
            S = align_f(eta)
            print(align_matrix_to_alignment(S,threshld), file=outf)

def test_alignment_bpe(align_f, concept, fs_test, es_test, out_filename, threshld, src_idx, trg_idx):
    temp_file = 'align_temp_bpe.moses'
    test_alignment(align_f, concept, fs_test, es_test, temp_file, threshld)
    print('align finished')
    utils.converse_BPE_to_word(temp_file, src_idx, trg_idx, out_filename)
    print('convert bpe to aligndata finished')

def evaluate(align_f, concept, fs_test, es_test, true_label_file, threshld=0.5, bpe=True, src_idx=None, trg_idx=None):
    prediction_file = 'evaluate_temp.moses'
    if bpe:
        if not src_idx or not trg_idx:
            print('needs idx data')
            return
        test_alignment_bpe(align_f, concept, fs_test, es_test, prediction_file, threshld, src_idx, trg_idx)
    else:
        test_alignment(align_f, concept, fs_test, es_test, prediction_file, threshld)
    return utils.compute_aer(prediction_file, true_label_file)

def train_network(dataset, network, optimizer, epochs, batch=100):
    print('training ... ')
    for e in range(epochs):
        print('  epoch : ', e)
        for i in range(dataset.sentences_num):
            f = dataset.src_sentences[i]
            e = dataset.trg_sentences[i]
            
            network([f,e])
            network.backward()
            if (i+1) % batch == 0:
                optimizer.step()
                optimizer.zero_grad()
                network.embedding.re_compute_sum_exp_norm()
    print('OK! training finished ')
    

def start_by_step_e(dataset, concept, align_f, embedding_model, optimizer, batch=100):
    print('start by step e : ')
    for i in range(dataset.sentences_num):
        f = dataset.src_sentences[i]
        e = dataset.trg_sentences[i]
        embedding_model((f,e))
        eta = concept.eta_U(f,e)
        alpha = align_f(eta)
        embedding_model.backward(th.Tensor(alpha))
        if (i+1) % batch == 0:
            print(i)
            optimizer.step()
            optimizer.zero_grad()
            network.embedding.re_compute_sum_exp_norm()
    print('OK! start by step e finished : ')

def load_fasttext_embedding_model(f_fasttext_model_file, e_fasttext_model_file):
    if os.path.exists(f_fasttext_model_file):
        f_model = fasttext.load_model(f_fasttext_model_file)
    if os.path.exists(e_fasttext_model_file):
        e_model = fasttext.load_model(e_fasttext_model_file)

    embedding_model = embedding.FastTextBPEEmbedding(f_model, e_model)
    return embedding_model
        





def main():

    # f_fasttext_vec_file = 'wiki.en.align.vec'
    # e_fasttext_vec_file = 'wiki.ro.align.vec'
    f_fasttext_model_file = 'en-ro-bpe-16K-en.bin'
    e_fasttext_model_file = 'en-ro-bpe-16K-ro.bin'

    f_fasttext_model_file = 'en_embedding_fasttext_model.bin'
    e_fasttext_model_file = 'ro_embedding_fasttext_model.bin'

    concept_count_v2 = 'weights/en-ro-bpe-count-p_concept-v2.weight'
    # concept_viterbi_v2 = 'weights/en-ro-viterbi-p_concept-v2.weight'
    # concept_activeset_v2 = 'weights/en-ro-activeset-p_concept-v2.weight'
    concept_init_weight_file = concept_viterbi_v2
    concept = Concept(get_pair_hashcode)
    concept.load_p_concept(concept_init_weight_file)

    emalgo = EmAlgo()
    active_set = ActiveSet(20)


    train_dataset = AlignDataset(f_train_filename, e_train_filename)
    test_dataset = AlignDataset(f_test_filename, e_test_filename)


    # f_pretrained_model = load_vectors(f_fasttext_vec_file)
    # e_pretrained_model = load_vectors(e_fasttext_vec_file)
    # embedding_model = FastTextPretrainedEmbedding(f_pretrained_model, e_pretrained_model)

    # embedding_model = load_fasttext_embedding_model(f_fasttext_model_file, e_fasttext_model_file)
    # embedding_model = BertEmbedding()
    embedding_model = load_fasttext_embedding_model(f_fasttext_model_file, e_fasttext_model_file)

    #embedding_model = FastTextEmbedding(f_model, e_model)
    network = embedding.Align(embedding_model)
    sgd = SGD(network.parameters(), lr=0.0005)

    score_before_init = evaluate(emalgo.align_wedding, 
                                 concept, 
                                 test_dataset.src_sentences, 
                                 test_dataset.trg_sentences, 
                                 true_label_file, 
                                 threshld=0.1,
                                 bpe=True,
                                 src_idx=src_bpe_idx_data,
                                 trg_idx=trg_bpe_idx_data)
    print('wedding concept score before init: ', score_before_init)

    score_before_init = evaluate(emalgo.align_wedding, 
                                 embedding_model,
                                 test_dataset.src_sentences, 
                                 test_dataset.trg_sentences, 
                                 true_label_file, 
                                 threshld=0.1,
                                 bpe=True,
                                 src_idx=src_bpe_idx_data,
                                 trg_idx=trg_bpe_idx_data)
    print('wedding embedding score before init : ', score_before_init)

    start_by_step_e(train_dataset, concept, emalgo.align_wedding, embedding_model, sgd, 100)
    
    score_after_init = evaluate(active_set.align_sparsemap, 
                                embedding_model,
                                test_dataset.src_sentences, 
                                test_dataset.trg_sentences, 
                                true_label_file, 
                                threshld=0.2,
                                bpe=True,
                                src_idx=src_bpe_idx_data,
                                trg_idx=trg_bpe_idx_data)
    print('sparsemap embedding after step e : ', score_after_init)

    sgd = SGD(network.parameters(), lr=0.0001)
    train_network(train_dataset, network, sgd, epoch=2, batch=100)

    thresholds = [i*0.1 for i in range(1,11)]
    score_after_trainings = []
    for thr in thresholds:

        score_after_training = evaluate(active_set.align_sparsemap, 
                                    embedding_model,
                                    test_dataset.src_sentences, 
                                    test_dataset.trg_sentences, 
                                    true_label_file, 
                                    threshld=thr,
                                    bpe=True,
                                    src_idx=src_bpe_idx_data,
                                    trg_idx=trg_bpe_idx_data)
        score_after_trainings.append(score_after_training)

    print('score after training : ', score_after_trainings)
    print('best score after training : ', np.min(score_after_trainings))

    
    out_parameter_file = 'embedding_weight.txt'
    np.savetxt('out_parameter_file' , embedding_model.weight.detach().numpy())
    print('embedding weight saved to : ', out_parameter_file)



if __name__ == '__main__':
    main()







