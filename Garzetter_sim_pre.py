import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import codecs
from torch.utils.data import DataLoader
import math
from gensim.models import KeyedVectors

from torch.utils.data import TensorDataset
from datetime import datetime
from bigramProb import createBigramModel
import random
import argparse
from Model import CNN, BiLSTM,AttentionCNN,C_LSTM,C_LSTMAttention
import os
import sys
import psutil
torch.manual_seed(1)
def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

'''write a list of place names into file '''
def write_place(file_name, place_names):
    f= open(file_name,"w+")
    for neg in place_names:
        temp= ''
        for negraw in neg:
            temp = temp+negraw+' ' 
        f.write(temp+'\n')
    f.close()

'''load the word embedding from a file'''
def load_embeding(emb_file):
    vectors = []
    words = []
    idx = 0
    word2idx = {}
    with open(emb_file, 'rb') as f:
        for l in f:
            line = l.decode().split()
            if len(line) > 5:
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                vect = np.array(line[1:]).astype(np.float)
                vectors.append(vect)
                emb_dim = len(vect)
    glove = {w: vectors[word2idx[w]] for w in words}
    return glove,emb_dim


'''extract the manually defined structural features of training samples'''
def extract_hc_feat(training_data,  hc_file):
    word_hcfs = {}
    for data in training_data:
        s_l = len(data)
        for i, word in enumerate(data):
            if word not in word_hcfs:
                word_hcfs[word] = np.zeros(6)
            word_hcfs[word][0]= word_hcfs[word][0]+1
            word_hcfs[word][1] = word_hcfs[word][1] + s_l
            word_hcfs[word][2] = word_hcfs[word][2] + i+1
            if s_l ==1:
                word_hcfs[word][3] = 1
            if i == 0:
                word_hcfs[word][4] = word_hcfs[word][4]+1
            if i == s_l-1:
                word_hcfs[word][5] = word_hcfs[word][5]+1
    for word in word_hcfs.keys():
        word_hcfs[word][1] = word_hcfs[word][1]/float(word_hcfs[word][0])
        word_hcfs[word][2] = word_hcfs[word][2]/float(word_hcfs[word][0])
        word_hcfs[word][4] = word_hcfs[word][4]/float(word_hcfs[word][0])
        word_hcfs[word][5] = word_hcfs[word][5]/float(word_hcfs[word][0])       
        word_hcfs[word][0] = word_hcfs[word][0]/float(len(training_data))
        
    # save to file

    f= open(hc_file,"w+")
    for word in word_hcfs.keys():
        temp = word +' '
        for raw in word_hcfs[word]:
            temp = temp+ str(raw)+' ' 
        f.write(temp+'\n')
    f.close()
    return word_hcfs

'''get the map of <word, id> and < id, word> from file containing the training examples'''
def extract_wid(pos_f,word_to_ix_n,ix_to_word_n,target_vocab_n):
    for line in open(pos_f):
        line = line.strip()
        if len(line) == 0:
            continue
        sentence = line.split(' ')
        for j, word in enumerate(sentence):
            if word not in word_to_ix_n:
                len_word_ix = len(word_to_ix_n)
                word_to_ix_n[word] = len_word_ix
                ix_to_word_n[len_word_ix] = word
                target_vocab_n.append(word)
    return word_to_ix_n,ix_to_word_n,target_vocab_n

''' load training examples from file and convert the word to word id '''
def extract_feat(pos_f,tag,x_train_n, y_train_n,hc_feats_before_n,word_to_ix_n,ix_to_word_n,target_vocab_n,listOfProb,START_WORD,flex_feat_len,max_char_len,max_len,pad_index):
    postive_num = 0
    for line in open(pos_f):
        line = line.strip()
        if len(line) == 0:
            continue
        sentence = line.split(' ')
        sen_len = len(sentence)
        if sen_len > max_len:
            del sentence[0:sen_len-max_len]
        postive_num = postive_num + 1
        y_train_n.append(tag)
        temp_s = []
        temp_hc = []
        for j, word in enumerate(sentence):
            if word not in word_to_ix_n:
                len_word_ix = len(word_to_ix_n)
                word_to_ix_n[word] = len_word_ix
                ix_to_word_n[len_word_ix] = word
                target_vocab_n.append(word)
            temp_s.append(word_to_ix_n[word])
            temp_hc.append(len(sentence))
            temp_hc.append(j+1)
            if j==0:
                pre_word = START_WORD
            else:
                pre_word = sentence[j-1]
            try:
                temp_hc.append(listOfProb[(word, pre_word)])
            except KeyError:
                temp_hc.append(0)
            if flex_feat_len > 3:
                char_loc_feat = feat_char_loc(word, max_char_len)
                temp_hc.extend(char_loc_feat)
        # pad setence with pad_index
        pad_num = max_len-len(temp_s)
        for i in range(0,pad_num):
            temp_s.insert(0,pad_index)
            temp_hc.insert(0,0)
            temp_hc.insert(0,0)
            temp_hc.insert(0,0)

            if flex_feat_len > 3:
                temp_hc.extend([0]*2*max_char_len)
        x_train_n.append(temp_s)
        hc_feats_before_n.append(temp_hc)
    return x_train_n, y_train_n,hc_feats_before_n,word_to_ix_n,ix_to_word_n,target_vocab_n,postive_num

'''load the training data from positive file ( positive examples) and negative file ( negative examples).
to improve the effenciency of loading the negative examples, the negative file was splited into mulitple smaller
 file with each contraining at most 10 million examples    '''
def load_training_data_os(pos_f, neg_f, START_WORD, bigram_file, hc_file,PAD, pad_index, flex_feat_len, max_char_len, max_len, bool_hc, split_l, ratio=1.0):
    s_index = []
    word_to_ix = {}
    ix_to_word = {}
    x_train = []
    y_train = []
    word_to_ix[PAD]=0
    ix_to_word[0] = PAD
    target_vocab = [PAD]
    hc_feats_before = []    
    postive_num = 0
    pos_training_data = []
    with codecs.open(pos_f, 'r',encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split(' ')
            pos_training_data.append(tokens)
    print('pos memory:' +str(sys.getsizeof(pos_training_data)/(1024.0*1024.0)))
    print(len(pos_training_data))
    listOfProb = createBigramModel(pos_training_data, START_WORD, bigram_file)
    word_hcfs = extract_hc_feat(pos_training_data,hc_file)
    print('listOfProb:' +str(sys.getsizeof(listOfProb)/(1024.0*1024.0)))
    print('word_hcfs:' +str(sys.getsizeof(word_hcfs)/(1024.0*1024.0)))
    pos_training_data = []
    '''search positive samples to create trainable samples'''
    neg_exe_str = 'wc -l ' + neg_f
    negative_num = int(os.popen(neg_exe_str).read().split()[0]) 
    split_c = int(negative_num / float(split_l))+1
    for i in range(split_c):
        print(str(i)+'...')
        if i < 26:
            neg_f_i = neg_f[0:len(neg_f)-4] + 'a' + chr(i+97)
        else:
            neg_f_i = neg_f[0:len(neg_f)-4] + 'b' + chr(i-26+97)
           
        word_to_ix,ix_to_word,target_vocab = extract_wid(neg_f_i,word_to_ix,ix_to_word,target_vocab)

    print('neg done')
    print('neg num:'+str(negative_num))
    
    '''The positive examples are loaded to the memory'''

    x_train, y_train,hc_feats_before,word_to_ix,ix_to_word,target_vocab,postive_num = extract_feat(pos_f,1,x_train, \
                                               y_train, hc_feats_before,word_to_ix,ix_to_word,target_vocab,listOfProb,\
                                                START_WORD,flex_feat_len,max_char_len,max_len,pad_index)
    y_train = np.array(y_train)
    x_train = np.array(x_train)
    hc_feats_before = np.array(hc_feats_before)       
    print('post done')
    random_num = int(negative_num*ratio)
    random_indexs = [random.randint(0, postive_num-1) for x in range(random_num)]
    s_index = random_indexs
    return split_c, x_train, y_train, s_index, word_hcfs, hc_feats_before, word_to_ix,ix_to_word,target_vocab,listOfProb

"""Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8"""
def binary_accuracy(preds, y):
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc, rounded_preds


def feat_char_loc(word, max_char_len):
    return_char_loc_vector = []
    for i in range(max_char_len-len(word)):
        return_char_loc_vector.append(0)
        return_char_loc_vector.append(0)
    for i, c in enumerate(word):
        return_char_loc_vector.append(i+1)
        return_char_loc_vector.append(len(word)-i)
    return return_char_loc_vector


def main():
    # parse parameters
    time_str = datetime.now().strftime('%m%d%H%M%S')
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--epoch', type=int, default=8)
    parser.add_argument('--train-batch-size', type=int, default=1000)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--model', type=int, default=1)
    parser.add_argument('--count', type=int, default=0)
    parser.add_argument('--osm_word_emb', type=int, default=1)
    parser.add_argument('--hc', type=int, default=1)
    parser.add_argument('--postive', type=int, default=12)
    parser.add_argument('--negative', type=int, default=12)
    parser.add_argument('--osmembed', type=int, default=2)
    parser.add_argument('--preloadsize', type=int, default=1000000)
    parser.add_argument('--filter_l', type=int, default=3)
    parser.add_argument('--split_l', type=int, default=10000000)
    parser.add_argument('--max_cache', type=int, default=13)
    parser.add_argument('--atten_dim', type=int, default=80)
    parser.add_argument('--filter_option', type=int, default=1)
    parser.add_argument('--cnn_hid', type=int, default=120)
    parser.add_argument('--optim', type=int, default=1)

    args = parser.parse_args()
    print ('epoch: '+str(args.epoch))
    print ('train batch size: '+str(args.train_batch_size))
    print ('test batch size: '+str(args.test_batch_size))
    print ('model: '+str(args.model))
    print ('count: '+str(args.count))
    print ('osm_word_emb: '+str(args.osm_word_emb))
    print ('hc: '+str(args.hc))
    print ('positive: '+str(args.postive))
    print ('negative: '+str(args.negative))
    print ('osmembed: '+str(args.osmembed))
    print ('preloadsize: '+str(args.preloadsize))
    print ('split_l: '+str(args.split_l))
    print ('max_cache: '+str(args.max_cache))
    print ('atten_dim: '+str(args.atten_dim))
    print ('filter_option: '+str(args.filter_option))
    print ('filter_l: '+str(args.filter_l))
    print ('cnn_hid: '+str(args.cnn_hid))
    print ('optim: '+str(args.optim))

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    START_WORD = 'start_string_taghu'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print (device)
    bool_mb_gaze = args.osm_word_emb
    bool_hc = args.hc
    lstm_layer_num = 1
    pos_f = 'data/positive'+str(args.postive)+'.txt'
    neg_f = 'data/negative'+str(args.negative)+'.txt' 
    bigram_file = 'data/'+ time_str+'-bigram.txt'
    hc_file = 'data/'+ time_str+'-hcfeat.txt'
    PAD = 'paddxk'
    pad_index = 0
    max_char_len = 20
    max_len = 20
    flex_feat_len = 3
    split_c, x_train_pos, y_train_pos, data_index_pos, word_hcfs, hc_feats_before_pos, word_to_ix, ix_to_word, target_vocab,listOfProb = \
                          load_training_data_os(pos_f,neg_f,START_WORD, bigram_file, hc_file,PAD,pad_index,flex_feat_len,max_char_len,max_len,bool_hc,args.split_l)#
    pos_unit = args.split_l
    print('the number of unique words are:'+str(len(word_to_ix)))
    vocab_file = 'data/'+ time_str+'vocab.txt'
    with open(vocab_file, 'w+',encoding='utf-8') as f:
         for word in word_to_ix.keys():
             f.write(word+' '+str(word_to_ix[word])+'\n')
         f.close()
    print('vocab file successfully saved:')        
    # map words to its word embeding
    glove_emb_file = 'data/glove.6B.50d.txt'
    glove, emb_dim = load_embeding(glove_emb_file)
    gazetteer_emb_file = 'data/osm_vector'+str(args.osmembed)+'.txt'

    if bool_mb_gaze:
        gazetteer_emb,gaz_emb_dim = load_embeding(gazetteer_emb_file)
    else:
        gaz_emb_dim = 0
    if bool_hc: 
        hc_len = 6
    else:
        hc_len = 0 
    matrix_len = len(word_to_ix)
    weight_dim = emb_dim+gaz_emb_dim+hc_len
    print('weight_dim: ' + str(weight_dim))
    print('entity_dim: ' + str(max_len*weight_dim))
    weights_matrix = np.zeros((matrix_len, weight_dim)) 
    words_found = 0
    
    for i, word in enumerate(target_vocab):
        try: 
            temp_glove = glove[word]
            words_found += 1
            temp_ofv = []
        except KeyError:
            temp_glove = np.random.normal(scale=0.6, size=(emb_dim,))
            temp_ofv = []
        if bool_mb_gaze:
            try: 
                temp_gaz = gazetteer_emb[word]
            except KeyError:
                temp_gaz = np.random.normal(scale=0.6, size=(gaz_emb_dim,))
        else:
            temp_gaz = []
        if bool_hc:
            try:
                temp_hc = word_hcfs[word]
            except KeyError:
                temp_hc = np.zeros(hc_len)
        else:
            temp_hc = []
        weights_matrix[i] = np.concatenate((temp_glove,temp_gaz,temp_hc, temp_ofv), axis=None)
    segment_lens = []
    garz_osm_pos_len = 0    
    garz_osm_pos_len += len(temp_glove)
    final_filter_w = []
    final_filter_w.append([])
    if bool_mb_gaze:
    	garz_osm_pos_len += len(temp_gaz)
    segment_lens.append(garz_osm_pos_len)
    if bool_hc:
        segment_lens.append(len(temp_hc)+flex_feat_len)
        final_filter_w.append([1,1,1])
    weights_matrix= torch.from_numpy(weights_matrix)
    tag_to_ix = {"p": 0, "n": 1}
    
    if args.model == 5:
        FILTER_SIZES = []
        for i in range(len(segment_lens)):
            if args.filter_option:
                FILTER_SIZES = [1,2,3]
            else:
                FILTER_SIZES = [2,3,4]
    else:
        if args.filter_option:
            FILTER_SIZES = [1,2,3]
        else:
            FILTER_SIZES = [2,3,4]
    OUTPUT_DIM = 1
    DROPOUT = 0.5
    HIDDEN_DIM = args.cnn_hid
    f_rec_name = 'experiments/record'+str(HIDDEN_DIM)+time_str+'.txt'
    f_record = open(f_rec_name, 'w+',encoding='utf-8')
    print (HIDDEN_DIM)
    print("lstm layer: {0}".format(lstm_layer_num), file=f_record)
    print("epoch number: {0}".format(args.epoch), file=f_record)
    print("train batch size: {0}".format(args.train_batch_size), file=f_record)
    print("max len: {0}".format(max_len), file=f_record)
    print("hidden: {0}".format(HIDDEN_DIM), file=f_record)
    """create a model"""
    if args.model == 1:
        model_path = 'model/lstm_model_'+str(HIDDEN_DIM)+time_str
        model = BiLSTM(weights_matrix, len(tag_to_ix), HIDDEN_DIM, lstm_layer_num,flex_feat_len).to(device)
        criterion = nn.CrossEntropyLoss()
    elif args.model == 2:
        model_path = 'model/atten_cnn_model_'+time_str
        model = AttentionCNN(weights_matrix, HIDDEN_DIM, FILTER_SIZES, OUTPUT_DIM, flex_feat_len, DROPOUT).to(device)
        criterion =  nn.BCEWithLogitsLoss()
    elif args.model == 3:
        fileter_l = args.filter_l
        model_path = 'model/clstm_model_'+time_str
        model = C_LSTM(weights_matrix, HIDDEN_DIM, fileter_l, args.atten_dim, len(tag_to_ix), flex_feat_len, DROPOUT).to(device)
        criterion = nn.CrossEntropyLoss()
    elif args.model == 4:
        fileter_l = args.filter_l
        model_path = 'model/clstm_attention_model_'+time_str
        model = C_LSTMAttention(weights_matrix, HIDDEN_DIM, fileter_l, True, args.atten_dim, OUTPUT_DIM, flex_feat_len, DROPOUT).to(device)
        criterion =  nn.BCEWithLogitsLoss()
    else:
        model_path = 'model/cnn_model_'+time_str
        model = CNN(weights_matrix, HIDDEN_DIM, FILTER_SIZES, OUTPUT_DIM, flex_feat_len, DROPOUT).to(device)
        criterion =  nn.BCEWithLogitsLoss()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("numer of trainable parameters: {0}".format(pytorch_total_params), file=f_record)
    print('numer of trainable parameters: ',str(pytorch_total_params))
    if not args.optim:
        optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(),lr=0.001)
    index_list= random.sample(range(split_c), split_c)
    train_size = math.ceil(0.8 * split_c)
    train_idx, test_idx = index_list[:train_size+1], index_list[train_size+1:]
    if args.max_cache > split_c:
        max_cache_size = split_c
    else:
        max_cache_size = args.max_cache
    cache_data = []
    cache_index = []
    
    for epoch in range(
            args.epoch):  # again, normally you would NOT do 300 epochs, it is toy data
        model.train()
        """ negative examples are too big thus we only save partial data in the memory as cache 
        and read the others directely from the negative file as needed"""
        for idx in train_idx:
            print(str(idx)+'...')
            if idx < 26:
                neg_f_i = neg_f[0:len(neg_f)-4] + 'a' + chr(idx+97)
            else:
                neg_f_i = neg_f[0:len(neg_f)-4] + 'b' + chr(idx-26+97)
            x_train_neg = []
            y_train_neg = []
            hc_feats_before_neg = []
            if idx not in cache_index:
                x_train_neg, y_train_neg, hc_feats_before_neg, word_to_ix,ix_to_word,target_vocab,cur_neg_num = extract_feat(neg_f_i,0,x_train_neg, \
                                           y_train_neg, hc_feats_before_neg,word_to_ix,ix_to_word,target_vocab,listOfProb,\
                                            START_WORD,flex_feat_len,max_char_len,max_len,pad_index)
                if len(cache_index) < max_cache_size:
                    cache_index.append(idx)
                    cache_data.append([x_train_neg, y_train_neg, hc_feats_before_neg,cur_neg_num ])
            else:
                c_idx = cache_index.index(idx)
                x_train_neg = cache_data[c_idx][0]
                y_train_neg = cache_data[c_idx][1]
                hc_feats_before_neg = cache_data[c_idx][2]
                cur_neg_num = cache_data[c_idx][3]
            pos_st_idx = idx*pos_unit
            pos_en_idx = idx*pos_unit + cur_neg_num
            x_train = [x_train_pos[j] for j in data_index_pos[pos_st_idx:pos_en_idx]]
            x_train.extend(x_train_neg)
            y_train = [y_train_pos[j] for j in data_index_pos[pos_st_idx:pos_en_idx]]
            y_train.extend(y_train_neg)
            hc_feats_before = [hc_feats_before_pos[j] for j in data_index_pos[pos_st_idx:pos_en_idx]]
            hc_feats_before.extend(hc_feats_before_neg)
            r_index_list= random.sample(range(len(y_train)), len(y_train))

            x_train = np.array(x_train)
            y_train = np.array(y_train)
            hc_feats_before = np.array(hc_feats_before)

            loop_count = math.ceil(len(x_train)/args.preloadsize)
            for loop in range(loop_count):
                if (loop+1)*args.preloadsize > len(x_train):
                    last_idx = len(x_train)
                else:
                    last_idx = (loop+1)*args.preloadsize
                x_tr = torch.tensor(x_train[r_index_list[loop*args.preloadsize:last_idx]], dtype=torch.long)
                y_tr = torch.tensor(y_train[r_index_list[loop*args.preloadsize:last_idx]], dtype=torch.float)
                hc_tr = torch.tensor(hc_feats_before[r_index_list[loop*args.preloadsize:last_idx]], dtype=torch.float)
                train = TensorDataset(x_tr,y_tr,hc_tr)
                trainloader = DataLoader(train, batch_size=args.train_batch_size,pin_memory=True,num_workers = 4)
                for sentence, tags, hcs in trainloader:
                    sentence, tags = sentence.to(device), tags.to(device)
                    hcs = hcs.view(len(sentence),max_len,flex_feat_len).to(device)
                    model.zero_grad()
                    predictions = model(sentence,hcs)
                    if args.model== 1:
                        loss = criterion(predictions, tags.squeeze().long())
                    elif args.model== 2:
                        loss = criterion(predictions, tags.unsqueeze(1))     
                    elif args.model == 3:
                        loss = criterion(predictions, tags.squeeze().long())
                    else:
                        loss = criterion(predictions, tags.unsqueeze(1))
                    loss.backward()
                    optimizer.step()
        correct = 0
        incorrect_place = []
        model.eval()
        test_sample_size = 0
        
        #test
        for idx in test_idx:
            print(str(idx)+'...')
            if idx < 26:
                neg_f_i = neg_f[0:len(neg_f)-4] + 'a' + chr(idx+97)
            else:
                neg_f_i = neg_f[0:len(neg_f)-4] + 'b' + chr(idx-26+97)

            x_train_neg = []
            y_train_neg = []
            hc_feats_before_neg = []
            if idx not in cache_index:
                x_train_neg, y_train_neg, hc_feats_before_neg, word_to_ix,ix_to_word,target_vocab,cur_neg_num = extract_feat(neg_f_i,0,x_train_neg, \
                                           y_train_neg, hc_feats_before_neg,word_to_ix,ix_to_word,target_vocab,listOfProb,\
                                            START_WORD,flex_feat_len,max_char_len,max_len,pad_index)
                if len(cache_index) < max_cache_size:
                    cache_index.append(idx)
                    cache_data.append([x_train_neg, y_train_neg, hc_feats_before_neg,cur_neg_num ])
            else:
                c_idx = cache_index.index(idx)
                x_train_neg = cache_data[c_idx][0]
                y_train_neg = cache_data[c_idx][1]
                hc_feats_before_neg = cache_data[c_idx][2]
                cur_neg_num = cache_data[c_idx][3]
                
            pos_st_idx = idx*pos_unit
            pos_en_idx = idx*pos_unit + cur_neg_num
            x_train = [x_train_pos[j] for j in data_index_pos[pos_st_idx:pos_en_idx]]
            x_train.extend(x_train_neg)
            y_train = [y_train_pos[j] for j in data_index_pos[pos_st_idx:pos_en_idx]]
            y_train.extend(y_train_neg)
            hc_feats_before = [hc_feats_before_pos[j] for j in data_index_pos[pos_st_idx:pos_en_idx]]
            hc_feats_before.extend(hc_feats_before_neg)
            r_index_list= random.sample(range(len(y_train)), len(y_train))
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            hc_feats_before = np.array(hc_feats_before)
            test_sample_size += len(x_train)
            loop_count = math.ceil(len(x_train)/args.preloadsize)
            for loop in range(loop_count):
                if (loop+1)*args.preloadsize > len(x_train):
                    last_idx = len(x_train)
                else:
                    last_idx = (loop+1)*args.preloadsize                
                x_test = torch.tensor(x_train[r_index_list[loop*args.preloadsize:last_idx]], dtype=torch.long)
                y_test = torch.tensor(y_train[r_index_list[loop*args.preloadsize:last_idx]], dtype=torch.float)
                hc_test = torch.tensor(hc_feats_before[r_index_list[loop*args.preloadsize:last_idx]], dtype=torch.float)
                test = TensorDataset(x_test, y_test,hc_test)
                testloader = DataLoader(test, batch_size=args.test_batch_size,pin_memory=True,num_workers = 4)
                for data, tags, hcs in testloader:
                    data, tags = data.to(device), tags.to(device)
                    hcs = hcs.view(len(data),max_len,flex_feat_len).to(device)
                    output = model(data,hcs)
                    if args.model == 1:
                        _, preds_tensor = torch.max(output, 1)
                    elif args.model== 2:
                        _, preds_tensor = torch.max(output, 1)
                    elif args.model == 3:
                        preds_tensor = output.argmax(dim=1)
                    else:
                        preds_tensor = torch.round(torch.sigmoid(output)).squeeze(1)
    
                    correct += sum(preds_tensor.eq(tags).cpu().numpy())
                    data,tags = data.to('cpu'), tags.to('cpu')
                    for i, se in enumerate(data):
                        ixs = se.numpy()
                        cur_s = []
                        for w in ixs:
                            if w:
                                cur_s.append(ix_to_word[w])
                        cur_s.append(str(tags[i].item()))
                        if not preds_tensor[i].eq(tags[i]):
                            incorrect_place.append(cur_s)
        incorrect_file = 'experiments/h'+ str(HIDDEN_DIM)+'epoch'+str(epoch)+time_str+'inc.txt'
        write_place(incorrect_file, incorrect_place)
        #write_place(correct_file, correct_place)
        print("test accuracy: %f", correct/test_sample_size)
        print("test accuracy: {0}".format(correct/test_sample_size),file=f_record)
        torch.save(model.state_dict(), model_path+'epoch'+str(epoch)+'.pkl')
    f_record.close()
if __name__ == '__main__':
    main()
