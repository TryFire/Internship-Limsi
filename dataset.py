import random
class AlignDataset():
    
    def __init__(self, src_file, trg_file, description=None):
        self.src_sentences = None
        self.trg_sentences = None
        self.sentences_num = None
        self.description = description
        self.read_file(src_file, trg_file)
        
    
    def read_file(self, src_file, trg_file):
        print('reading files...')
        f_file = open(src_file)
        self.src_sentences = list(map(lambda x:x.strip().split(),f_file.readlines()))
        f_file.close()
        
        e_file = open(trg_file)
        self.trg_sentences = list(map(lambda x:x.strip().split(),e_file.readlines()))
        e_file.close()   
        print('reading finished.')
        
        self.sentences_num = len(self.src_sentences)

    def shuffle(self):
        c = list(zip(self.src_sentences, self.trg_sentences))
        random.shuffle(c)
        a,b = zip(*c)
        self.src_sentences, self.trg_sentences = list(a), list(b)

    
    def detail(self):
        from itertools import chain
        lensen = len(self.src_sentences)
        lenss = list(map(len, self.src_sentences))
        maxsen = max(lenss)
        minsen = min(lenss)
        totallen = sum(lenss)//lensen
        totoalwords = len(set(chain.from_iterable(self.src_sentences)))
        print('src : ')
        print('  sentences:     ', lensen)
        print('  max sentence:  ', maxsen)
        print('  min sentence:  ', minsen)
        print('  mean sentence: ', totallen)
        print('  total words  : ', totoalwords)
        print('  max words:     ', max(map(len, chain.from_iterable(self.src_sentences))))
        
        ensen = len(self.trg_sentences)
        lenss = list(map(len, self.trg_sentences))
        maxsen = max(lenss)
        minsen = min(lenss)
        totallen = sum(lenss)//lensen
        totoalwords = len(set(chain.from_iterable(self.trg_sentences)))
        print('trg : ')
        print('  sentences:     ', lensen)
        print('  max sentence:  ', maxsen)
        print('  min sentence:  ', minsen)
        print('  mean sentence: ', totallen)
        print('  total words  : ', totoalwords)
        print('  max words:     ', max(map(len, chain.from_iterable(self.trg_sentences))))

