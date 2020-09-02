import numpy as np 
import sys, argparse, os

def generate_command_parameters():
    parser = argparse.ArgumentParser(
        description='transofrm bpe vecfile to vecfile which can be used in vecmap.')
    parser.add_argument(
        '--vec', dest='vecfile',
        type=str, help='vector file')
    parser.add_argument(
        '-n', dest='nwords',default=16000,
        type=int, help='number of bpe pieces')
    parser.add_argument(
        '--dim', dest='dim',default=300,
        type=int, help='embedding dimension')

    return parser.parse_args()


def transofrm(vec_file, outfile, nwords, dim):

    words_vec = np.loadtxt(vec_file, delimiter=',')
    
    outf = open(outfile, 'w+')
    print('%d %d'%(nwords, dim),file=outf)

    for i in range(nwords):
        print('%d %s'%(i, np.array2string(words_vec[i, :])[1:-1]), file=outf)


args = generate_command_parameters()

vec_file = args.vecfile
outf = 'tomap_%s'%(vec_file.split('/')[-1])
nwords = args.nwords
dim = args.dim

print('transforming ... ')
transofrm(vec_file, outf, nwords, dim)
print('OK! transform finished!')



