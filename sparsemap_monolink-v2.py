from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment


import io

from activeset import ActiveSet
from embedding import *
from dataset import AlignDataset
from emalgos import EmAlgo
from concept import Concept
import utils 

import numpy as np
import pickle
import os 

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


f_language = 'en'
e_language = 'ro'

data_folder = '/vol/work2/2017-NeuralAlignments/exp-xinneng/en-ro-bpe'


f_train_filename = '%s/train.merg.en-ro.cln.en.utf8.low.lenSent50'%(data_folder)
e_train_filename = '%s/train.merg.en-ro.cln.ro.utf8.low.lenSent50'%(data_folder)
f_test_filename = '%s/corp.test.ro-en.cln.en.low'%(data_folder)
e_test_filename = '%s/corp.test.ro-en.cln.ro.low'%(data_folder)

bpe_folder = '/vol/work2/2017-NeuralAlignments/exp-xinneng/en-ro-bpe'

src_bpe_file = '%s/train.merg.en-ro.cln.en.utf8.low.lenSent50-16000.bpe'%(bpe_folder)
trg_bpe_file = '%s/train.merg.en-ro.cln.ro.utf8.low.lenSent50-16000.bpe'%(bpe_folder)
src_bpe_test_file = '%s/corp.test.ro-en.cln.en.low-16000.bpe'%(bpe_folder) 
trg_bpe_test_file = '%s/corp.test.ro-en.cln.ro.low-16000.bpe'%(bpe_folder)
src_bpe_idx_data = '%s/corp.test.ro-en.cln.en.low-16000.idx'%(bpe_folder)
trg_bpe_idx_data = '%s/corp.test.ro-en.cln.ro.low-16000.idx'%(bpe_folder)

f_train_filename = src_bpe_file
e_train_filename = trg_bpe_file
f_test_filename = src_bpe_test_file
e_test_filename = trg_bpe_test_file



# f_embedding_model_file = '%s_embedding_fasttext_model.bin'%(f_language)
# e_embedding_model_file = '%s_embedding_fasttext_model.bin'%(e_language)



true_label_file = '%s/test.en-ro.ali.startFrom1'%(data_folder)

train_dataset = AlignDataset(f_train_filename, e_train_filename)
test_dataset = AlignDataset(f_test_filename, e_test_filename)

dataset.detail()
emalgo = EmAlgo()
concept = Concept(get_pair_hashcode)

weight_file = 'en-de-count-init-p-concept.weight'
if os.path.exists(weight_file):
    print(' load init weight file')
    concept.load_p_concept(weight_file)
else:
    print(' init weight file count')
    emalgo.init_p_concep_by_count_pair(concept, train_dataset)
    print(' init weight file finished')
    concept.save_p_concept(weight_file)


score_before_init = evaluate(emalgo.align_wedding, 
                             concept, 
                             test_dataset.src_sentences, 
                             test_dataset.trg_sentences, 
                             true_label_file, 
                             threshld=0.1,
                             bpe=True,
                             src_idx=src_bpe_idx_data,
                             trg_idx=trg_bpe_idx_data)
print('wedding concept score before train: ', score_before_init)


diffs = emalgo.train(train_dataset, concept, emalgo.align_wedding)
concept.save_p_concept('en-de-viterbi-monolink-p-concept.weight')

score_before_init = evaluate(emalgo.align_wedding, 
                             concept, 
                             test_dataset.src_sentences, 
                             test_dataset.trg_sentences, 
                             true_label_file, 
                             threshld=0.1,
                             bpe=True,
                             src_idx=src_bpe_idx_data,
                             trg_idx=trg_bpe_idx_data)
print('wedding concept score after train: ', score_before_init)

active_set = ActiveSet(30)
diffs = emalgo.train(dataset, concept, active_set.solve_sparse_map)

trained_weight_file = 'en-de-spasemap-monolink-v2-using-viterbi-p-concept.weight'
concept.save_p_concept(trained_weight_file)
print(diffs)
        

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

print('sparsemap score after training : ', score_after_trainings)
print('sparsemap best score after training : ', np.min(score_after_trainings))


