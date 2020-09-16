from Model import CNN,BiLSTM, C_LSTM,AttentionCNN,C_LSTMAttention
from Garzetter_sim_pre import load_embeding
from Garzetter_sim_pre import feat_char_loc
import json, re
import torch
from utility import replace_digs,hasNumbers
import numpy as np
from core import extract_sim
from gensim.models import KeyedVectors 
from datetime import datetime
import argparse
from wordsegment import segment

''' return the intersetion set of two lists'''
def interset(list1,list2):
    return_list = []
    for l1 in list1:
        try:
            index = list2.index(l1)
            return_list.append(l1)
            list2.remove(index)
        except ValueError:
            continue
    return return_list

def interset_adv(list1,list2):
    first_place = ''
    second_place = ''
    for i in list1:
        first_place += i
    for j in list2:
        second_place += j
    if first_place==second_place:
        match = 1
    else:
        match = 0
    return match

''' get the TP and FP value given the ground truth and predicted result'''
def interset_num(list1,list2,detected_place_names,place_names):
    TP = 0
    FP = 0
    FN_list = [1]*len(place_names)
    place_detect_score = [0]*len(place_names)
    for i, l1 in enumerate(list1):
        bool_ins = False
        for j, l in enumerate(list2):
            if l1 == l:
                place_detect_score[j] = 1  
                TP += 1
                FN_list[j] = 0
                bool_ins = True
            else:
                if (l1[0] >= l[0] and l1[0] <= l[1]) or (l1[1] >= l[0] and l1[1] <= l[1]) or (l[0] >= l1[0] and l[0] <= l1[1]) or (l[1] >= l1[0] and l[1] <= l1[1]):
                    pen = interset_adv(list(detected_place_names[i]),list(place_names[j]))
                    if pen != 1:
                        pen = 0.5
                    else:
                        TP += pen
                    FP += (1-pen)
                    FN_list[j] = (1-pen)
                    place_detect_score[j] = pen  
                    bool_ins = True
        if not bool_ins:
            FP += 1
    return TP,FP,sum(FN_list), place_detect_score


def load_word_index(index_file):
    word2idx = {}
    max_char_len = 0
    with open(index_file, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            if len(word) > max_char_len:
                max_char_len = len(word)
            if len(line)== 1:
                word2idx['noword'] = int(line[0])
            else:
                word2idx[line[0]] = int(line[1])
    return word2idx, max_char_len

'''load the bigram model from file'''
def load_bigram_model(bigram_file):
    bigram_model = {}

    with open(bigram_file, 'rb') as f:
        for l in f:
            line = l.decode().split()
            if len(line) == 3:
                bigram_model[(line[0],line[1])] = float(line[2])
    return bigram_model

''' judge if s is the sub list of l ''' 
def is_Sublist(l, s):
    sub_set = False
    if s == []:
        sub_set = True
    elif s == l:
        sub_set = True
    elif len(s) > len(l):
        sub_set = False
    else:
        for i in range(len(l)):
            if len(l)-i < len(s):
                return False
            if l[i] == s[0]:
                n = 1
                while (n < len(s)) and (l[i+n] == s[n]):
                    n += 1
                if n == len(s):
                    sub_set = True
                    return sub_set
    return sub_set

''' judge if two list has shared elements  '''
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

''' get sub list of a list with the sub list length below max_lem'''
def sub_lists(list1,max_len): 
    # store all the sublists  
    sublist = []
    sublist_index = []   
    # first loop  
    for i in range(len(list1) + 1):         
        # second loop  
        for j in range(i + 1, len(list1) + 1):             
            # slice the subarray
            if j-i<max_len:
                sub = list1[i:j]
                sub_index = range(i,j)
                sublist_index.append(list(sub_index))
                
                sublist.append(sub)
    return sublist,sublist_index

''' get the embedding of a sentence  '''
def sentence_embeding(sentence, trained_emb, word_idx,glove_emb,osm_emb,\
                      max_len,emb_dim,gaz_emb_dim,\
                      max_char_len,bool_mb_gaze,\
                      PAD_idx,START_WORD,listOfProb, char_hc_emb,flex_feat_len):
    matrix_len = len(sentence)
    weights_matrix = np.zeros((max_len, emb_dim+gaz_emb_dim+6+flex_feat_len)); 

    for i in range(0,max_len-matrix_len):
        char_loc_feat = []
        if flex_feat_len - 3 > 0:
            char_loc_feat = [0]*(flex_feat_len - 3)
        weights_matrix[i] = np.concatenate((trained_emb[PAD_idx],[0,0,0],char_loc_feat),axis=None)
    for i, word in enumerate(sentence):
        temp_hc = []
        temp_hc.append(len(sentence))
        temp_hc.append(i+1)
        if i==0:
            pre_word = START_WORD
        else:
            pre_word = sentence[i-1]
        try:
            temp_hc.append(listOfProb[(word, pre_word)])
        except KeyError:
            temp_hc.append(0)
        if flex_feat_len - 3 > 0:
            char_loc_feat = feat_char_loc(word, max_char_len)
            temp_hc.extend(char_loc_feat)
        if word not in word_idx.keys():
            try: 
                temp_glove = glove_emb[word]
            except KeyError:
                temp_glove = np.random.normal(scale=0.1, size=(emb_dim,))
            if bool_mb_gaze:
                try: 
                    temp_gaz = osm_emb[word]
                except KeyError:
                    temp_gaz = np.random.normal(scale=0.1, size=(gaz_emb_dim,))
            else:
                temp_gaz = []
            try:
                temp_hc6 = char_hc_emb[word]
            except KeyError:
                temp_hc6 = np.random.normal(scale=2, size=6)
            weights_matrix[i+max_len-matrix_len]=np.concatenate((temp_glove,temp_gaz,temp_hc6,np.asarray(temp_hc)
), axis=None)
        else:
            w_idx = word_idx[word]
            weights_matrix[i+max_len-matrix_len]=np.concatenate((trained_emb[w_idx],np.asarray(temp_hc)),axis=None)

    return weights_matrix

def main():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--model', type=int, default=3)
    parser.add_argument('--thres', type=float, default=0.82)
    parser.add_argument('--model_ID', type=str, default= '0224234050')
    parser.add_argument('--osmembed', type=int, default= 1)
    parser.add_argument('--osm_word_emb', type=int, default=1)
    parser.add_argument('--hc', type=int, default=1)
    parser.add_argument('--hidden', type=int, default=100)
    parser.add_argument('--bool_embed', type=int, default=0)
    parser.add_argument('--region', type = int, default=1)
    parser.add_argument('--atten_dim', type = int, default=80)
    parser.add_argument('--epoch', type = int, default=0)
    parser.add_argument('--filter', type = int, default=1)
    parser.add_argument('--filter_l', type = int, default=3)
    parser.add_argument('--bool_remove', type = int, default=1)

    args = parser.parse_args()
    print ('model: '+str(args.model))
    print ('thres: '+str(args.thres))
    print ('model_ID: '+args.model_ID)
    print ('osmembed: '+str(args.osmembed))
    print ('osm_word_emb: '+str(args.osm_word_emb))
    print ('hc: '+str(args.hc))
    print ('hidden: '+str(args.hidden))
    print ('atten_dim: '+str(args.atten_dim))
    print ('epoch: '+str(args.epoch))
    print ('filter: '+str(args.filter))
    print ('filter_l: '+str(args.filter_l))
    print ('bool_remove: '+str(args.bool_remove))

    model_type = args.model
    postive_pro_t = args.thres
    PAD_idx = 0
    s_max_len = 18
    bool_mb_gaze = args.osm_word_emb
    model_ID = args.model_ID;
    gazetteer_emb_file = 'data/osm_vector'+str(args.osmembed)+'.txt'
    bigram_file = 'data/'+model_ID+'-bigram.txt'
    hcfeat_file = 'data/'+model_ID+'-hcfeat.txt'
    START_WORD = 'start_string_taghu'
    bigram_model = load_bigram_model(bigram_file)

    if bool_mb_gaze:
        gazetteer_emb,gaz_emb_dim = load_embeding(gazetteer_emb_file)
    else:
        gazetteer_emb = []
        gaz_emb_dim = 0
    char_hc_emb,_ = load_embeding(hcfeat_file)
    word_idx_file = 'data/'+model_ID+'vocab.txt'
    word2idx, max_char_len = load_word_index(word_idx_file)
    max_char_len = 20
    if args.bool_embed==1:
        glove_emb_file = 'model/GoogleNews-vectors-negative300.bin'
        emb_model = KeyedVectors.load_word2vec_format(glove_emb_file, binary=True)
        glove_emb = emb_model.wv
        emb_dim = len(emb_model.wv['the'])
    elif args.bool_embed==2:
        glove_emb_file = 'data/glove.6B.100d.txt'
        glove_emb, emb_dim = load_embeding(glove_emb_file)
    else:
        glove_emb_file = 'data/glove.6B.50d.txt'
        glove_emb, emb_dim = load_embeding(glove_emb_file)
    weight_l = emb_dim+gaz_emb_dim+6
    weights_matrix = np.zeros((len(word2idx.keys()), weight_l))
    weights_matrix= torch.from_numpy(weights_matrix)
    tag_to_ix = {"p": 0, "n": 1}
    HIDDEN_DIM = args.hidden
    lstm_layer_num = 2
    model_path = 'model/'+model_ID+'epoch'+str(args.epoch)+'.pkl'
    OUTPUT_DIM = 1
    DROPOUT = 0.5
    flex_feat_len = 3
    if args.filter:
        FILTER_SIZES = [1,3,5]
    else:
        FILTER_SIZES = [2,3,4]
        
    if model_type == 1:
        model = BiLSTM(weights_matrix, len(tag_to_ix), HIDDEN_DIM, lstm_layer_num)
    elif model_type == 2:
        model = AttentionCNN(weights_matrix, HIDDEN_DIM, FILTER_SIZES, OUTPUT_DIM, flex_feat_len, DROPOUT)        
    elif model_type == 3:
        fileter_l = args.filter_l
        model = C_LSTM(weights_matrix, HIDDEN_DIM, fileter_l, args.atten_dim, len(tag_to_ix), flex_feat_len, DROPOUT)
    elif model_type == 4:
        fileter_l = args.filter_l
        model = C_LSTMAttention(weights_matrix, HIDDEN_DIM, fileter_l, True, args.atten_dim, OUTPUT_DIM , flex_feat_len, DROPOUT)
    else:
        model = CNN(weights_matrix, HIDDEN_DIM, FILTER_SIZES, OUTPUT_DIM, flex_feat_len,DROPOUT)
    model.load_state_dict(torch.load(model_path,map_location='cpu'))
    model.eval()
    if model_type == 1:
        np_word_embeds = model.word_embeds.weight.detach().numpy()
    else:
        np_word_embeds = model.embedding.weight.detach().numpy() 
    index_t = 0
    time_str = datetime.now().strftime('%m%d%H%M%S')
    raw_result_file = 'data/cnn_result'+time_str+'.txt'
    result_file = 'data/place_garze'+time_str+'.txt'
    save_file = open(raw_result_file,'w') 
    place_garze = open(result_file,'w')
    true_count = 0
    TP_count = 0
    FP_count = 0
    FN_count = 0
    if args.region==1:
        t_json_file = "data/houston_floods_2016_annotations.json"#"data/raw_tweet.txt"
    elif args.region==2:
        t_json_file = "data/chennai_floods_2015_annotations.json"#"
    else:
        t_json_file = "data/louisiana_floods_2016_annotations.json"#"
    with open(t_json_file) as json_file:
        js_data = json.load(json_file)
        for key in js_data.keys():
            tweet = js_data[key]['text']
            place_names = []
            place_offset = []
            print (key+': '+tweet)
            for cur_k in js_data[key].keys():
                if cur_k == 'text':
                    tweet = js_data[key][cur_k]
                else:
                    if js_data[key][cur_k]['type'] != 'ambLoc':
                        row_nobrackets = re.sub("[\(\[].:;*?[\)\]]", "", js_data[key][cur_k]['text'])         
                        corpus = [word.lower() for word in re.split("[. #,&\"\',’]",row_nobrackets)]
                        corpus = [word  for word in corpus if word]
                        corpus = [replace_digs(word) for word in corpus]
                        place_names.append(tuple(corpus))
                        place_offset.append(tuple([int(js_data[key][cur_k]['start_idx']),int(js_data[key][cur_k]['end_idx'])-1]))
            last_remove = ['area','region']
            true_count += len(place_names)
            detected_place_names = []
            detected_offsets = []
            save_file.write('#'*50)
            save_file.write('\n')
            save_file.write(key+': '+tweet+'\n')
            place_lens = {}
            detected_score = {}
            sentences, offsets = extract_sim(tweet,word2idx.keys())
            print('*'*50)
            print(tweet)
            if sentences:
                print(sentences)
            for idx, sentence in enumerate(sentences):
                if sentence:
                    cur_off = offsets[idx]
                    index_t += 1
                    all_sub_lists,sub_index = sub_lists(sentence,s_max_len)
        
                    input_emb = np.zeros((len(all_sub_lists),s_max_len,emb_dim+gaz_emb_dim+6+flex_feat_len))
                    for i, sub_sen in enumerate(all_sub_lists):
                        input_emb[i] = sentence_embeding(sub_sen, np_word_embeds,word2idx,glove_emb,\
                                                  gazetteer_emb,s_max_len,emb_dim,\
                                                  gaz_emb_dim,max_char_len,bool_mb_gaze,\
                                                 PAD_idx,START_WORD,bigram_model,char_hc_emb,flex_feat_len)
                    input_emb= torch.from_numpy(input_emb).float()
                    if model_type==1:
                        output = model.predict(input_emb)
                        _, preds_tensor = torch.max(output, 1)
                        pos_prob = output.detach().numpy()[:,1]
                    elif model_type == 2:
                        tem_output = model.predict(input_emb)
                        pos_prob = torch.sigmoid(tem_output).detach().numpy()
                        pos_prob = pos_prob.reshape(pos_prob.shape[0])
                        preds_tensor = torch.round(torch.sigmoid(tem_output)).squeeze(1).detach()
                    elif model_type == 3:
                        output = model.predict(input_emb)
                        _, preds_tensor = torch.max(output, 1)
                        pos_prob = torch.sigmoid(output).detach().numpy()
                        pos_prob = pos_prob[:,1]
                    elif model_type == 4:
                        tem_output = model.predict(input_emb)
                        pos_prob = torch.sigmoid(tem_output).detach().numpy()
                        pos_prob = pos_prob.reshape(pos_prob.shape[0])
                        preds_tensor = torch.round(torch.sigmoid(tem_output)).squeeze(1).detach()
                    else:
                        tem_output = model.core(input_emb)
                        pos_prob = torch.sigmoid(tem_output).detach().numpy()
                        pos_prob = pos_prob.reshape(pos_prob.shape[0])
                        preds_tensor = torch.round(torch.sigmoid(tem_output)).squeeze(1).detach()        
                    preds = -preds_tensor.numpy()
                    postives = []

                    for i, p in enumerate(preds):
                         if pos_prob[i] > postive_pro_t:
                             postives.append(i)
                    origin_pos_prob = pos_prob        
                    pos_prob = pos_prob[postives]
                    sort_index = (-pos_prob).argsort()
                    selected_sub_sen = []
                    for index in sort_index:
                        if not selected_sub_sen:
                            selected_sub_sen.append(postives[index])
                        else:
                            temp_sub_sen = selected_sub_sen.copy()
                            bool_added = True
                            for p in temp_sub_sen:
                                if  intersection(sub_index[p], sub_index[postives[index]]) and \
                                            not (is_Sublist(sub_index[postives[index]],sub_index[p])):
                                    bool_added = False
                                    break
                            if bool_added:
                                selected_sub_sen.append(postives[index])    
                    final_sub_sen = selected_sub_sen.copy()
                    for i in selected_sub_sen:
                        for j in selected_sub_sen:
                            if not (i==j):
                                if is_Sublist(sub_index[j], sub_index[i]):
                                    final_sub_sen.remove(i)
                                    break
                    for i in final_sub_sen:
                        if args.bool_remove:
                            if all_sub_lists[i][-1] in last_remove:
                                del all_sub_lists[i][-1]
#
                        detected_place_names.append(tuple(all_sub_lists[i]))
                        detected_offsets.append(tuple([cur_off[sub_index[i][0]][0],cur_off[sub_index[i][-1]][1]]))
                        save_file.write(str(round(origin_pos_prob[i],3))+':'+str(all_sub_lists[i])+'\n')
                        place_garze.write(str(round(origin_pos_prob[i],3))+':'+str(all_sub_lists[i])+'\n')
                        print(str(round(origin_pos_prob[i],3))+':'+str(all_sub_lists[i]))
            return_set = interset(detected_offsets,place_offset)
            c_tp, c_fp,c_fn, place_detect_score = interset_num(detected_offsets,place_offset,detected_place_names,place_names)
            print('c_tp:'+str(c_tp)+' c_fp:'+str(c_fp)+' c_fn:'+str(c_fn))
            for p, i in enumerate(place_names):
                cur_len_p = 0
                for pp in i:
                    if hasNumbers(pp):
                        groups = re.split('(\d+)',pp)
                        groups = [x for x in groups if x]
                        cur_len_p += len(groups)
                    else:
                        segments = segment(pp)
                        cur_len_p += len(segments)
                if cur_len_p in place_lens.keys():
                    place_lens[cur_len_p] += 1
                    detected_score[cur_len_p] += place_detect_score[p]
                else:
                    place_lens[cur_len_p] = 1
                    detected_score[cur_len_p] = place_detect_score[p]
            print(place_names)
            print(return_set)
            TP_count += c_tp
            FP_count += c_fp
            FN_count += c_fn
    P = TP_count/(TP_count+FP_count) 
    R = TP_count/(TP_count+FN_count) 
    F = (2*P*R) / (P+R)
    print('recall:' + str(R))
    print('precision:' + str(P))
    print('f1:' +  str(F))
    print('TP:' + str(TP_count))
    print('FP:' + str(FP_count))
    print('FN:' + str(FN_count))

    print('true count:' + str(true_count))
    save_file.close()
    place_garze.close()
if __name__ == '__main__':
    main()
