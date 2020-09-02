
from transformers import BertModel, BertTokenizer
import numpy as np

import torch as th
from torch.nn.parameter import Parameter

from sklearn.metrics.pairwise import cosine_similarity

from activeset import ActiveSet
import torch.nn.functional as f

class Embedding(th.nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()

    
    def get_embed_list(self, sent_pair: list) -> np.ndarray:
        raise "not defined"
    
    def get_similarity(self, X1 : np.ndarray, X2 : np.ndarray) -> np.ndarray:
        similarity = X1 @ self.weight.detach().numpy() @ (X2.T)
        return similarity
#        return th.Tensor((cosine_similarity(X1, X2) + 1.0) / 2.0)
        
    def normalize(self):
        self.weight = Parameter(f.normalize(self.weight))
    
    def apply_distortion(self, sim_matrix: np.ndarray, ratio: float = 0.5) -> np.ndarray:
        shape = sim_matrix.shape
        if (shape[0] < 2 or shape[1] < 2) or ratio == 0.0:
            return sim_matrix

        pos_x = np.array([[y / float(shape[1] - 1) for y in range(shape[1])] for x in range(shape[0])])
        pos_y = np.array([[x / float(shape[0] - 1) for x in range(shape[0])] for y in range(shape[1])])

        distortion_mask = -abs((pos_x - np.transpose(pos_y))) * ratio

        return nsim_matrix + distortion_mask



    
    def eta_U(self, f, e, distortion=False):
        data = [f,e]
        src_embedding, trg_embedding = self.get_embed_list(data)

        data_save = {}

        scale = min(len(src_embedding), len(trg_embedding))
        # src_indexs = [int(w) for w in f]
        # trg_indexs = [int(w) for w in f]

        data_save['scale'] = scale
        data_save['src_embedding'] = src_embedding
        data_save['trg_embedding'] = trg_embedding

        similarity = self.get_similarity(src_embedding, trg_embedding)

        self.save_for_backward(data_save)

        similarity = similarity - self.maxvalue - np.log(self.sum_exp_sim)
        if distortion:
            similarity = self.apply_distortion(sim)
        
        return similarity

    def re_compute_sum_exp_norm(self):
        temp = self.src_words_vec @ self.weight.detach().numpy() @  (self.trg_words_vec.T)
        self.maxvalue = np.max(temp)
        exp_value = np.exp( temp - self.maxvalue )
        self.sum_exp_sim = np.sum(exp_value)
        softmax_value = exp_value / self.sum_exp_sim 
        self.softmax_gradient = self.src_words_vec.T @ softmax_value @ self.trg_words_vec
        print(self.sum_exp_sim)
        #self.backword_softmax = th.Tensor(self.src_words_vec).T @ (th.Tensor(self.softmax_value) @ th.Tensor(self.trg_words_vec))
        
        
    def forward(self, data : list):
        f, e = data
        similarity = self.eta_U(f, e)
        return th.Tensor(similarity)
        
    
    def backward(self,g):
        data_save = self.saved_tensors
        temp = data_save['scale']*self.softmax_gradient 
        second = data_save['src_embedding'].T @ g.numpy() @ data_save['trg_embedding']

        
        self.weight.backward( th.Tensor(temp - second) )

        
    def save_for_backward(self, t):
        self.saved_tensors = t

class BertEmbedding(Embedding):
    def __init__(self, model='bert-base-multilingual-cased'):
        super(BertEmbedding, self).__init__()    
        models = {
            'bert-base-uncased': (BertModel, BertTokenizer),
            'bert-base-multilingual-cased': (BertModel, BertTokenizer),
            'bert-base-multilingual-uncased': (BertModel, BertTokenizer),
#             'xlm-mlm-100-1280': (XLMModel, XLMTokenizer),
#             'roberta-base': (RobertaModel, RobertaTokenizer),
#             'xlm-roberta-base': (XLMRobertaModel, XLMRobertaTokenizer),
#             'xlm-roberta-large': (XLMRobertaModel, XLMRobertaTokenizer),
        }
        
        model_class, tokenizer_class = models[model]
        self.bert_tokenizer = tokenizer_class.from_pretrained(model)
        self.bert_model = model_class.from_pretrained(model)
        
        self.embedding_dim = 768
        self.weight = Parameter(th.diag(th.ones(self.embedding_dim)))

        self.src_e = {}
        self.trg_e = {}
        
    def get_embed_list(self, sent_pair: list) -> np.ndarray:
        sent_ids = [self.bert_tokenizer.convert_tokens_to_ids(x) for x in sent_pair]
        inputs = [self.bert_tokenizer.prepare_for_model(sent, return_token_type_ids=True, return_tensors='pt')['input_ids'] for sent in sent_ids]
        outputs = [self.bert_model(in_ids) for in_ids in inputs]
        vectors = [x[0].detach().numpy()[0][1:-1] for x in outputs]
        
        return vectors

class FastTextEmbedding(Embedding):
    def __init__(self, src_model, trg_model):
        super(FastTextEmbedding, self).__init__()
        
        self.src_model = src_model
        self.trg_model = trg_model
        self.src_embedding_dim = src_model.get_dimension()
        self.trg_embedding_dim = trg_model.get_dimension()
        assert self.src_embedding_dim == self.trg_embedding_dim  
        
        self.weight = Parameter(th.diag(th.ones(self.src_embedding_dim)))
        
    def get_embed_list(self, sent_pair: list) -> np.ndarray:
        src_sentence, trg_sentence = sent_pair
        src_embedding = np.array([self.src_model.get_word_vector(w) for w in src_sentence])
        trg_embedding = np.array([self.trg_model.get_word_vector(w) for w in trg_sentence])
        
        return [src_embedding, trg_embedding]

class FastTextPretrainedEmbedding(Embedding):
    def __init__(self, src_model, trg_model):
        super(FastTextPretrainedEmbedding, self).__init__()
        self.src_model = src_model
        self.trg_model = trg_model

        self.embedding_dim = 300



        self.weight = Parameter(th.diag(th.ones(self.embedding_dim)))
        
    def get_embed_list(self, sent_pair: list) -> np.ndarray:
        src_sentence, trg_sentence = sent_pair

        src_embedding = np.array([self.src_model[w] if w in self.src_model else np.array(self.defaultvec) for w in src_sentence ])
        trg_embedding = np.array([self.trg_model[w] if w in self.trg_model else np.array(self.defaultvec)  for w in trg_sentence ])

        return [src_embedding, trg_embedding]

class FastTextBPEEmbedding(Embedding):
    def __init__(self, src_model_file, trg_model_file):
        super(FastTextBPEEmbedding, self).__init__()
        self.src_words_vec = np.loadtxt(src_model_file, delimiter=',')
        self.trg_words_vec = np.loadtxt(trg_model_file, delimiter=',')

        self.embedding_dim = self.src_words_vec.shape[1]

        print('  src vecs : ', len(self.src_words_vec))
        print('  trg vecs : ', len(self.trg_words_vec))

        self.weight = Parameter(th.diag(th.ones(self.embedding_dim)))

        self.re_compute_sum_exp_norm()

    def get_embed_list(self, sent_pair: list) -> np.ndarray:
        src_sentence, trg_sentence = sent_pair
        src_embedding = self.src_words_vec[list(map(int, src_sentence)), :]
        trg_embedding = self.trg_words_vec[list(map(int, trg_sentence)), :]
        
        return [src_embedding, trg_embedding]

        
class Align(th.nn.Module):
    def __init__(self, model, T=20):
        super(Align, self).__init__()
        self.embedding = model
        self.active_set = ActiveSet(T)
        
    
    def forward(self,data,distortion=True):
        similarity = self.embedding(data, distortion)
        alpha = self.active_set.align_sparsemap(similarity.data.numpy())
        self.lowbound = np.sum(alpha * similarity.data.numpy()) - self.active_set.divergence
        alpha = th.Tensor(alpha)
        self.save_for_backward(alpha)
        return alpha

    def backward(self,g=None):
        if g == None:
            alpha = self.saved_tensors
            self.embedding.backward(alpha)
        else:
            self.embedding.backward(g)
        
    def save_for_backward(self, t):
        self.saved_tensors = t


