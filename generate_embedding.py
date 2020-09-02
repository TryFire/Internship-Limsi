import fasttext

from tempfile import NamedTemporaryFile
import numpy as np 
import sys, argparse, os

def generate_command_parameters():
    parser = argparse.ArgumentParser(
        description='Using fasttext to generate the embedding vector.')
    parser.add_argument(
        '--src', dest='src_language', default='en',
        type=str, help='source language')
    parser.add_argument(
        '--trg', dest='trg_language', default='ro',
        type=str, help='target language')

    return parser.parse_args()


def fasttext_train_unsupervised(bpe_file, nwords, outf, dim=300,minCount=1,minn=10,maxn=10):
    words = set()
    with open(bpe_file,'r') as f:
        for line in f:
            for w in line.split():
                if w not in words:
                    words.add(w)
    print('training word vecs by fasttext')
    model = fasttext.train_unsupervised(bpe_file, dim=dim, minCount=minCount,minn=minn,maxn=maxn)
    print('OK! training finished')
    

    words_vec = np.zeros((nwords, dim))
    for i in range(nwords):
        if str(i) in words:
            words_vec[i,:] = model.get_word_vector(str(i))

    np.savetxt(outf, words_vec, delimiter=',')
    print('OK! model saved to %s '%(outf))


def main():
    args = generate_command_parameters()

    bpe_folder = '/workdir/xinneng/%s-%s-bpe'%(args.src_language, args.trg_language)

    src_train_en_ro_en = 'train.merg.en-ro.cln.en.utf8.low.lenSent50-16000.bpe'
    trg_train_en_ro_ro = 'train.merg.en-ro.cln.ro.utf8.low.lenSent50-16000.bpe'
    src_test_en_ro_en = 'corp.test.ro-en.cln.en.low-16000.bpe'
    trg_test_en_ro_ro = 'corp.test.ro-en.cln.ro.low-16000.bpe'


    src_train_en_de_en = 'corp.train.de-en.low.cln.en.final.lenSent50-16000.bpe'
    trg_train_en_de_de = 'tcorp.train.de-en.low.cln.de.final.lenSent50-16000.bpe'
    src_test_en_de_en = 'corp.test.de-en.en.low.ngoho-16000.bpe'
    trg_test_en_de_de = 'corp.test.de-en.de.low.ngoho-16000.bpe'

    if args.src_language == 'en' and args.trg_language =='ro':
        src_train_bpe_file = '%s/%s'%(bpe_folder,src_train_en_ro_en)
        trg_train_bpe_file = '%s/%s'%(bpe_folder,trg_train_en_ro_ro)
        src_test_bpe_file = '%s/%s'%(bpe_folder,src_test_en_ro_en)
        trg_test_bpe_file = '%s/%s'%(bpe_folder,trg_test_en_ro_ro)
    elif args.src_language == 'en' and args.trg_language =='de':
        src_train_bpe_file = '%s/%s'%(bpe_folder,src_train_en_de_en)
        trg_train_bpe_file = '%s/%s'%(bpe_folder,trg_train_en_de_de)
        src_test_bpe_file = '%s/%s'%(bpe_folder,src_test_en_de_en)
        trg_test_bpe_file = '%s/%s'%(bpe_folder,trg_test_en_de_de)
    else: return 

    nwords = 16000


    src_embedding_file = '%s-%s-bpe-16K-%s.txt'%(args.src_language, args.trg_language, args.src_language)
    trg_embedding_file = '%s-%s-bpe-16K-%s.txt'%(args.src_language, args.trg_language, args.trg_language)

    with NamedTemporaryFile('w+', encoding='utf-8') as src_bpe, \
         NamedTemporaryFile('w+', encoding='utf-8') as trg_bpe:

        with open(src_train_bpe_file,'r') as f:
            for line in f:
                print(line, file=src_bpe)

        with open(src_test_bpe_file,'r') as f:
            for line in f:
                print(line, file=src_bpe)

        with open(trg_train_bpe_file,'r') as f:
            for line in f:
                print(line, file=trg_bpe)

        with open(trg_test_bpe_file,'r') as f:
            for line in f:
                print(line, file=trg_bpe)

        src_bpe.flush()
        trg_bpe.flush()


        fasttext_train_unsupervised(src_bpe.name, nwords, src_embedding_file)
        fasttext_train_unsupervised(trg_bpe.name, nwords, trg_embedding_file)







