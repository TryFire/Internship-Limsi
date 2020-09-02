
from collections import Counter
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import numpy as np
def get_pair_hashcode(f,e):
    return "%s|||%s"%(f,e)
class EmAlgo():
    def __init__(self):
        self.count_concept = None
        
    def init_p_concep_by_count_pair(self,concept, dataset):
        if self.count_concept:
            concept.p_concept = self.count_concept
        else:
            self.count_concept = self.__compute_pair_occur_norm_count(dataset.src_sentences, dataset.trg_sentences, concept.get_pair_hashcode)
            concept.p_concept = self.count_concept
        
        
    def __compute_pair_occur_norm_count(self,src_sentences,trg_sentences,pair_hash):
        total_count = 0
        count = defaultdict(int)
        for src_s, trg_s in zip(src_sentences, trg_sentences):
            for src_word in set(src_s):
                for trg_word in set(trg_s):
                    count[pair_hash(src_word,trg_word)] += 1
                    total_count += 1
        for pair in count.keys():
            count[pair] /= total_count
        return count    
    
    def update_count_pair(self, p_count, sentence_pair, S):
        src_s, trg_s = sentence_pair
        for j in range(len(src_s)):
            for i in range(len(trg_s)):
                if S[j, i] != 0:
                    p_count[get_pair_hashcode(src_s[j], trg_s[i])] += S[j, i]
                        

    def compute_diff(self, pc1, pc2):
        sum_error = 0
        for key in pc1:
            sum_error += abs(pc1[key]-pc2[key])
        for key in pc2:
            sum_error += abs(pc1[key]-pc2[key])
        return sum_error/2    
                    
    def normalize_p_count(self, p_count):
        total_count = sum(p_count.values())
        for pair in p_count:
            p_count[pair] /= total_count

    
    def align_wedding(self, eta):
        cost = -eta
        # compute the best alignment by Hungarian wedding algorithm,
        row_ind, col_ind = linear_sum_assignment(cost)
        
        solution = np.zeros(cost.shape)
        for j,i in zip(row_ind, col_ind):
            solution[j,i] = 1
        return solution
    
    def train(self, dataset, concept, slove_alignment, iterations=10, threshold=0.001):
        diffs = []
        last_concept = concept.p_concept
        for ite in range(iterations):  
            print('iteration: ', ite)
            # init count of each pair(f_j,e_i) aligned to 0
            p_count = defaultdict(int)
            for pair_iter,(f,e) in enumerate(zip(dataset.src_sentences,dataset.trg_sentences)):
                if pair_iter%100000 == 0: print(' pair iter : ', pair_iter)
                eta = concept.eta_U(f,e)
                alignments = slove_alignment(eta)
                self.update_count_pair(p_count,(f,e),alignments)
                # alignments,proba = slove_alignment(f,e,concept)
                # compute count of each alignment
                # self.update_count_pair(p_count,(f,e),alignments, concept.word_empty, proba)
            # normalize
            self.normalize_p_count(p_count)
            # test converge
            diff = self.compute_diff(last_concept, p_count)
            diffs.append(diff)
            print('  diff:', diff)

            concept.update_p_concept(p_count)
            if diff < threshold:
                break
            last_concept = concept.p_concept
        return diffs

