import pickle
from collections import defaultdict

import numpy as np


class Concept:
    def __init__(self, pair_hashf, alpah=1, epsilon=1e-12):
        
        self.get_pair_hashcode = pair_hashf
        
        self.word_empty = '_W '
        
        self.p_concept = defaultdict(int)
        
        self.alpha = alpah
        self.epsilon = epsilon
        self.infin = -1e5
        self.distortion = distortion
    
    def eta_U(self, src_s, trg_s, pos_constraint=True):
        len_src_s, len_trg_s = len(src_s), len(trg_s)
#         etau = {}
#         for i, src_word in enumerate(src_s):
#             for j, trg_word in enumerate(trg_s):
#                 scale = abs(i/len_src_s-j/len_trg_s)*self.alpha if pos_constraint else 0
#                 etau[(i,j)] = np.log(self.p_concept[self.get_pair_hashcode(src_word, trg_word)]) - scale
#         return etau
        etau = np.zeros((len_src_s, len_trg_s))
        for j, src_word in enumerate(src_s):
            for i, trg_word in enumerate(trg_s):
                scale = abs(j/len_src_s-i/len_trg_s)*self.alpha if pos_constraint else 0
                if self.p_concept[self.get_pair_hashcode(src_word, trg_word)] <= 0:
                    etau[j,i] = self.infin - scale
                else: etau[j,i] = np.log(self.p_concept[self.get_pair_hashcode(src_word, trg_word)]+self.epsilon) - scale
        
        return etau
        
    def build_matrix_cost(self, src_s, trg_s, pos_constraint=True):
        return -self.eta_U(src_s, trg_s, pos_constrain)
        
    def update_p_concept(self,p_count):
        self.p_concept = p_count.copy()
        
    
    def load_p_concept(self, filename):
        print('Concept load weights from : ', filename)
        f_weight = open(filename, 'rb')
        self.p_concept = pickle.load(f_weight)
        f_weight.close() 
        print('OK! Concept loaded weights from : ', filename)    
        
    def save_p_concept(self,filename):
        print('Concept save weights to : ', filename)
        f_weight = open(filename, 'wb')
        pickle.dump(self.p_concept, f_weight)
        f_weight.close()
        print('OK! Concept saved weights to : ', filename)    


