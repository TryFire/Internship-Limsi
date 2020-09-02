import numpy as np
from collections import Counter

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



def converse_BPE_to_word(align_data, source_idx_data, target_idx_data, outfile):
    source_idx_batch = []
    target_idx_batch = []
    alignment_set = []

    align_file = open(align_data)
    align_lines = align_file.readlines()
    
    source_file = open(source_idx_data)
    source_lines = source_file.readlines()
    
    target_file = open(target_idx_data)
    target_lines = target_file.readlines()

    for i in  align_lines:
        alignment_set.append(i.split())
    
    for i in source_lines:
        source_idx_batch.append(i.split())

    for i in target_lines:
        target_idx_batch.append(i.split())
    
    word_alignments = []
    for subword_alignment, ref_s, ref_t in zip(alignment_set, source_idx_batch, target_idx_batch):
        word_alignment = []
        for subword in subword_alignment:
            subword_src = subword.split('-')[0]
            subword_tgt = subword.split('-')[1]
            word_src = 0
            word_tgt = 0
            for s in ref_s:
                word_src_ref = s.split('-')[0]
                subword_src_ref = s.split('-')[1]
                if subword_src == subword_src_ref:
                    word_src = word_src_ref
            for t in ref_t:
                word_tgt_ref = t.split('-')[0]
                subword_tgt_ref = t.split('-')[1]
                if subword_tgt == subword_tgt_ref:
                    word_tgt = word_tgt_ref
            if word_src + '-'+ word_tgt not in word_alignment:
                word_alignment.append(word_src + '-'+word_tgt)
        word_alignments.append(word_alignment)
        
    with open(outfile,'w',encoding='utf-8') as outf:
        for lineno in word_alignments:
            print(' '.join(lineno),file=outf)
        outf.flush()
        
    return word_alignments



