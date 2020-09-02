
import numpy as np 
from scipy.optimize import linear_sum_assignment

class ActiveSet():
    def __init__(self,T=10,epsilon=1e-9,stop_percent = 0.01,debug=False):
        self.T = T
        self.epsilon = epsilon
        self.debug = debug
        self.stop_percent = stop_percent

    def align_sparsemap(self, etau):
        len_src, len_trg = etau.shape[0], etau.shape[1]
        x_t = self.init_wedding(-etau)
        # return x_t
        S = [[x_t,1]]
        for iteration in range(self.T):
            if self.debug: print('iteration : ',iteration)
            grad = self._gradient(x_t, etau)
            s_t, d_fw = self._fw_direction(grad, x_t, len_src, len_trg)
            (index_v ,v_t), d_aw = self._aw_direction(grad, x_t, S)
            
            g_converge = self.multi(grad, d_fw, -1, 1)
            if self.debug: print('  is converge: ', g_converge)
            #print('  alpha : ', alpha.values())
        
            if self.multi(grad, d_fw, -1, 1) <= self.epsilon:
                S = [[x_t,1]]
                return self._active_set_to_alignment(S)
                
            
            if self.multi(grad, d_fw, neg_a=-1,neg_b=1) >= self.multi(grad, d_aw, neg_a=-1,neg_b=1):
                d_t, gamma_max = d_fw, 1
                FW_step = True
            else:
                #print('  a_v_t : ', alpha[v_t])
                #d_t, gamma_max = d_aw, alpha[v_t]/(1-alpha[v_t])
                d_t, gamma_max = d_aw, S[index_v][1]/(1-S[index_v][1])
                FW_step = False
            
            #print('  FW step : ', FW_step)
            #print('  gamma max : ', gamma_max)
                
            ## line search to get optimal gamma
            gamma_t = self.binary_line_search(x_t, d_t, gamma_max, etau)
            
            #print('  gamma t : ', gamma_t)
            
            ## update x_t
#             x_t1 = hashabledict()
#             for key in x_t.keys():
#                 x_t1[key] = x_t[key] + gamma_t*d_t[key]
#             x_t = x_t1

            lastxt = x_t.copy()
            x_t = x_t + gamma_t * d_t
            
            
            ## update alpha_v
            if FW_step:
                if gamma_t == 1:
                    S = [[s_t,1]]
                else:
                    S.append([s_t,gamma_t])
                    for i in range(len(S)-1):
                        S[i][1] = S[i][1]*(1-gamma_t)
                
#                 alpha[s_t] = alpha.get(s_t, 0)*(1-gamma_t) + gamma_t
                
#                 for key in alpha.keys():
#                     if key != s_t:
#                         alpha[key] *= (1-gamma_t)
                            
            else:
                for i in range(len(S)):
                    if i != index_v:
                        S[i][1] = S[i][1]*(1+gamma_t)
                    else:
                        S[i][1] = S[i][1]*(1+gamma_t) - gamma_t
                if gamma_t == gamma_max:
                    S.pop(index_v)
                else:
                    pass
                
            lastf = np.sum(lastxt*lastxt) - np.sum(lastxt*etau)
            curf = np.sum(x_t*x_t) - np.sum(x_t*etau)
            if abs((curf- lastf)/lastf) < self.stop_percent:
                return self._active_set_to_alignment(S)
                
                
                
#                 alpha[v_t] = alpha.get(v_t, 0)*(1+gamma_t) - gamma_t
#                 for key in alpha.keys():
#                     if key != v_t:
#                         alpha[key] *= (1+gamma_t)
        
        return self._active_set_to_alignment(S)
    
    def _active_set_to_alignment(self, S):
        self.divergence = np.sum([solution[1]*np.log(solution[1]) for solution in S])
        return sum([solution[0]*solution[1] for solution in S])
        
#         alignments = []
#         proba_align = []
#         for solution in S:
#             src_j, trg_i = [], []
#             for (j,i),a in solution.items():
#                 if a == 1:
#                     src_j.append(j)
#                     trg_i.append(i)
#             alignments.append((src_j, trg_i))
#             proba_align.append(alpha[solution])
#         return alignments,proba_align
            
            
            
            
    def binary_line_search(self, x_t, d_t, gamma_max, etau):
        if self.debug: print('  binary line searching ... ')
        def grad_gamma(gamma):
            u = x_t + gamma*d_t
            grad_u = self._gradient(u,etau)
            return self.multi(grad_u, d_t)
#             u = {}
#             for key in x_t.keys():
#                 u[key] = x_t[key] + gamma*d_t[key]
#             grad_u = self._gradient(u,etau)
#             return self.multi(grad_u, d_t)
        
        #print('  g when gamma is max : ', grad_gamma(gamma_max))
        
        if grad_gamma(gamma_max) <= 0:
            return gamma_max
        
        gamma_l, gamma_r = 0, gamma_max
        while True:
            gamma_m = (gamma_l + gamma_r) / 2
            grad = grad_gamma(gamma_m)
            #print('    in binary line search, grad is : ',grad)
            if grad == 0:
                return gamma_m
            elif abs(grad) <= 1e-8:
                return gamma_m
            elif grad < 0:
                gamma_l = gamma_m
            elif grad > 0:
                gamma_r = gamma_m
            
            
    
    def _fw_direction(self, grad, x_t, J, I):
        if self.debug: print('  compute FW direction...')
        cost = grad
        
        s_t = np.zeros((J,I))
        d_t = -x_t
        
        if self.debug : print(cost)
        
        row_ind, col_ind = linear_sum_assignment(cost)
        
        for j,i in zip(row_ind,col_ind):
            s_t[j,i] = 1
            d_t[j,i] += 1
        if self.debug: print(s_t)
        return s_t, d_t
        
#         cost = np.zeros((J,I))
#         d_t = {}
#         s_t = hashabledict()
#         for j in range(J):
#             for i in range(I):
#                 cost[j,i] = grad[(j,i)]
#                 s_t[(j,i)] = 0
#                 d_t[(j,i)] = -x_t[(j,i)]
                
#         row_ind, col_ind = linear_sum_assignment(cost)
#         #print(row_ind, col_ind)
#         for j,i in zip(row_ind,col_ind):
#             s_t[(j,i)] += 1
#             d_t[(j,i)] += 1
        
#         return s_t, d_t
        
        
    
    
    def _aw_direction(self, grad, x_t, S):
        if self.debug: print('  compute AW direction...')
        i = np.argmax([self.multi(grad,v) for v,prob in S])
        v_t = S[i][0]
        d_t = x_t - v_t
        return (i, v_t), d_t
#         i = np.argmax([self.multi(grad,v) for v in S])
#         v_t = S[i]
#         d_t = {}
#         for key in v_t.keys():
#             d_t[key] = x_t[key] - v_t[key]
            
#         return v_t, d_t
            
    
    def multi(self, a, b, neg_a=1, neg_b=1):
#         res = 0
#         for key in a.keys():
#             res += a[key]*b[key]*neg_a*neg_b
#         return res
        return np.sum((neg_a*a)*(neg_b*b))
                
                
            
        
    def _gradient(self, solution, etau):
#         grad = {}
#         for key in solution.keys():
#             grad[key] = solution[key] - etau[key]
#         return grad
        return solution - etau

    def init_wedding(self,cost):
        row,col = linear_sum_assignment(cost)
        solution = np.zeros(cost.shape)
        for j,i in zip(row, col):
            solution[j,i] = 1
        return solution
    def get_alignment_matrix(self, sim_matrix: np.ndarray):
        m, n = sim_matrix.shape
        forward = np.eye(n)[sim_matrix.argmax(axis=1)]  # m x n
        backward = np.eye(m)[sim_matrix.argmax(axis=0)]  # n x m
        return forward * backward.transpose()
        
    def _init_solution(self,src_s, trg_s):
#         solution = hashabledict()
#         for j, src_word in enumerate(src_s):
#             for i, trg_word in enumerate(trg_s):
#                 if j == i:
#                     solution[(j,i)] = 1
#                 else:
#                     solution[(j,i)] = 0
#         return solution
        solution = np.zeros((len(src_s),len(trg_s)))
        for j, src_word in enumerate(src_s):
            for i, trg_word in enumerate(trg_s):
                if j == i:
                    solution[j,i] = 1
                else:
                    solution[j,i] = 0
        return solution        
        
    