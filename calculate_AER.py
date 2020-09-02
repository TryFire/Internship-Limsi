# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 13:45:13 2018

@author: ngoho
"""

import numpy as np
from collections import Counter
import argparse


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('reference_alignment', help='Reference alignment, First word index is 1')
parser.add_argument('prediction_alignment', help='Predicted alignment')
parser.add_argument('--prediction_alignment_start_from', default=1,
                    help='If the first word index in predicted alignment is 0')

args = parser.parse_args()


alignment_set = open(args.prediction_alignment, encoding='utf8')
target_file = open(args.reference_alignment, encoding='utf8')
start_from = 1 - int(args.prediction_alignment_start_from)

def compute_aer(prediction_alignment, reference_alignment, prediction_alignment_start_from=1):
    alignment_set = open(prediction_alignment, encoding='utf8')
    target_file = open(reference_alignment, encoding='utf8')
    start_from = 1 - int(prediction_alignment_start_from)

    alignment_batch = []
    for line in alignment_set:
        align = line.split()
        new_align = []
        for a in align:
            c1 = a.split('-')[0]
            c2 = a.split('-')[1]
            #CHECK INDEX
            al = str(int(c1)+start_from) +'-'+ str(int(c2)+start_from)
            new_align.append(al)
        alignment_batch.append(new_align)


    target_lines = target_file.readlines()
    target_lines = [str(line[:-1]) for line in target_lines]
    target_lines = np.reshape(target_lines, (np.int(len(target_lines)/2), 2))

    sure_batch = []
    possible_batch = []

    sure = target_lines[:,0]
    possible = target_lines[:,1]

    for i in sure:
        sure_batch.append(i.split())

    for i in possible:
        possible_batch.append(i.split())

    AER = []
    sure_alignment = 0.
    possible_alignment = 0.
    count_alignment = 0.
    count_sure = 0.

    for sure, possible, alignment in zip(sure_batch, possible_batch, alignment_batch ):
        for w in alignment:
            if w in sure:
                sure_alignment+=1.
            if w in possible:
                possible_alignment+=1.

        count_alignment += float(len(alignment))
        count_sure += float(len(sure))

    return (1. - (sure_alignment*2 + possible_alignment)/ (count_alignment + count_sure) )


print(compute_aer(alignment_set, target_file, start_from))

