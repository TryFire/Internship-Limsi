import sys


source_idx_data = sys.argv[3]
target_idx_data = sys.argv[4]

align_data = sys.argv[1]
outfile = sys.argv[2]

def converse_BPE_to_word():
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
        
    return word_alignments

word_alignments = converse_BPE_to_word()

with open("%s.moses"%(outfile),'w',encoding='utf-8') as outf:
    for lineno in word_alignments:
        print(' '.join(lineno),file=outf)
    outf.flush()






