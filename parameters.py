
f_language = 'en'
e_language = 'ro'
bpe = True 
bpe_size = 16000
distortion = True 

if bpe:
    file_prefix = '%s-%s-dis-%s'%(f_language, e_language, str(distortion))
else:
    file_prefix = '%s-%s-bpe-%d-dis-%s'%(f_language, e_language, bpe_size, str(distortion))

if distortion:
    pass

en_de_data_folder = '/vol/work2/2017-NeuralAlignments/exp-xinneng/en-de-bpe'

en_de_en_train_filename = '%s/corp.train.de-en.low.cln.en.final.lenSent50'%(en_de_data_folder)
en_de_de_train_filename = '%s/corp.train.de-en.low.cln.de.final.lenSent50'%(en_de_data_folder)
en_de_en_test_filename = '%s/corp.test.de-en.en.low.ngoho'%(en_de_data_folder)
en_de_de_test_filename = '%s/corp.test.de-en.de.low.ngoho'%(en_de_data_folder)

en_de_true_filename = '%s/alignmentDeEn.fixed.ali.startFrom1.en-de.ngoho'%(en_de_data_folder)


en_de_bpe_folder = '/vol/work2/2017-NeuralAlignments/exp-xinneng/en-de-bpe'

en_de_en_bpe_train_filename = '%s/corp.train.de-en.low.cln.en.final.lenSent50-%d.bpe'%(en_de_bpe_folder, bpe_size)
en_de_de_bpe_train_filename = '%s/corp.train.de-en.low.cln.de.final.lenSent50-%d.bpe'%(en_de_bpe_folder, bpe_size)
en_de_en_bpe_test_filename = '%s/corp.test.de-en.en.low.ngoho-%d.bpe'%(en_de_bpe_folder, bpe_size)
en_de_de_bpe_test_filename = '%s/corp.test.de-en.de.low.ngoho-%d.bpe'%(en_de_bpe_folder, bpe_size)
en_de_en_bpe_idx_data = '%s/corp.test.de-en.en.low.ngoho-%d.idx'%(en_de_bpe_folder, bpe_size)
en_de_de_bpe_idx_data = '%s/corp.test.de-en.de.low.ngoho-%d.idx'%(en_de_bpe_folder, bpe_size)




en_ro_data_folder = '/vol/work2/2017-NeuralAlignments/exp-xinneng/en-ro-bpe'

en_ro_en_train_filename = '%s/train.merg.en-ro.cln.en.utf8.low.lenSent50'%(en_ro_data_folder)
en_ro_ro_train_filename = '%s/train.merg.en-ro.cln.ro.utf8.low.lenSent50'%(en_ro_data_folder)
en_ro_en_test_filename = '%s/corp.test.ro-en.cln.en.low'%(en_ro_data_folder)
en_ro_ro_test_filename = '%s/corp.test.ro-en.cln.ro.low'%(en_ro_data_folder)

en_ro_true_filename = '%s/test.en-ro.ali.startFrom1'%(en_ro_data_folder)


en_ro_bpe_folder = '/vol/work2/2017-NeuralAlignments/exp-xinneng/en-ro-bpe'

en_ro_en_bpe_train_filename = '%s/train.merg.en-ro.cln.en.utf8.low.lenSent50-%d.bpe'%(en_ro_bpe_folder, bpe_size)
en_ro_ro_bpe_train_filename = '%s/train.merg.en-ro.cln.ro.utf8.low.lenSent50-%d.bpe'%(en_ro_bpe_folder, bpe_size)
en_ro_en_bpe_test_filename = '%s/corp.test.ro-en.cln.en.low-%d.bpe'%(en_ro_bpe_folder, bpe_size) 
en_ro_ro_bpe_test_filename = '%s/corp.test.ro-en.cln.ro.low-%d.bpe'%(en_ro_bpe_folder, bpe_size)
en_ro_en_bpe_idx_data = '%s/corp.test.ro-en.cln.en.low-%d.idx'%(en_ro_bpe_folder, bpe_size)
en_ro_ro_bpe_idx_data = '%s/corp.test.ro-en.cln.ro.low-%d.idx'%(en_ro_bpe_folder, bpe_size)



if f_language == 'en' and e_language == 'ro':
    true_label_file = en_ro_true_filename
    if bpe == True:
        f_train_filename = en_ro_en_bpe_train_filename
        e_train_filename = en_ro_ro_bpe_train_filename
        f_test_filename = en_ro_en_bpe_test_filename
        e_test_filename = en_ro_ro_bpe_test_filename
        f_idx_data = en_ro_en_bpe_idx_data
        e_idx_data = en_ro_ro_bpe_idx_data
    else:
        f_train_filename = en_ro_en_train_filename
        e_train_filename = en_ro_ro_train_filename
        f_test_filename = en_ro_en_test_filename
        e_test_filename = en_ro_ro_test_filename
elif f_language == 'en' and e_language == 'de':
    true_label_file = en_de_true_filename
    if bpe == True:
        f_train_filename = en_de_en_bpe_train_filename
        e_train_filename = en_de_de_bpe_train_filename
        f_test_filename = en_de_en_bpe_test_filename
        e_test_filename = en_de_de_bpe_test_filename
        f_idx_data = en_de_en_bpe_idx_data
        e_idx_data = en_de_de_bpe_idx_data
    else:
        f_train_filename = en_de_en_train_filename
        e_train_filename = en_de_de_train_filename
        f_test_filename = en_de_en_test_filename
        e_test_filename = en_de_de_test_filename




#### 

training_batch_size = 500
training_learning_rate = 0.00005
training_epochs = 5







