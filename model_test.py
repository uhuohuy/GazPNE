#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 21:40:54 2021

@author: hu_xk
"""
from Model import CNN,BiLSTM, C_LSTM,AttentionCNN,C_LSTMAttention
from Gazetteer_weight import load_embeding
from Gazetteer_weight import feat_char_loc
import json, re
import torch
from utility import *
import numpy as np
from core import extract_sim
from gensim.models import KeyedVectors 
from datetime import datetime
import argparse
from wordsegment import segment
from CMUTweetTagger import runtagger_parse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F

WORD_POS = 0
TAG_POS = 1
noun_tags_tweet = ['N','$','^','A', 'O','G'] #'P',
nan_single = ['A', 'O']
not_start = ['P']
not_end = ['P','A']
IGNORE = [] #'#','NEWLINE'

                 
def extract_nouns_tweet(terms_arr,max_len):
    new_arr = []
    for item in terms_arr:
        new_item = list(item)
        if item[2] < 0.4 or item[0].lower()=='of':
            new_item[1] = noun_tags_tweet[0]
        new_arr.append(new_item)
    span_arr, noun_array, indexs = generate_nouns(new_arr)
    return_list = []
    return_list_index = []
    return_pos = []
    for i, sublist in enumerate(noun_array):
        cur_list, cur_index = sub_lists_pos_adv(sublist,indexs[i], new_arr, max_len)
        return_list.extend(cur_list)
        return_list_index.extend(cur_index)
        for c_index in cur_index:
            return_pos.append([new_arr[index][1] for index in c_index])

    return return_list_index, return_list,return_pos

def generate_nouns(terms_arr):
    size = len(terms_arr)
    span_arr = []
    indexs = []
    i = 0
    return_results = []
    while (i < size):
        term_info = terms_arr[i]
        if (term_info[TAG_POS] in noun_tags_tweet and term_info[TAG_POS] not in not_start):
            skip = gen_sentence_raw(terms_arr,i)
            if skip==1 and term_info[TAG_POS] in nan_single:
                i += 1
                span_arr.append(0)
            else:
                i +=  skip
                temp = []
                for j in range(skip):
                    span_arr.append(1)
                    temp.append(terms_arr[i-skip+j][WORD_POS])
    #                temp_pos.append(terms_arr[i-skip+j][TAG_POS])
                return_results.append(temp)
    #            pos_result.append(temp_pos)
                indexs.append(list(range(i-skip,i)))
        else:
            i += 1
            span_arr.append(0)
    return span_arr, return_results, indexs#, pos_result


def sub_lists_pos_adv(list1, list_index, tags, max_len): 
    sublist = []
    sublist_index = []   
    # first loop  
    for i in range(len(list1) + 1):         
        # second loop  
        for j in range(i + 1, len(list1) + 1):             
            # slice the subarray
            if j-i<max_len:
                if tags[list_index[i]][TAG_POS] not in not_start \
                and  tags[list_index[j-1]][TAG_POS] not in not_end and \
                not (j-i==1 and tags[list_index[i]][TAG_POS] in nan_single):
                    sub = list1[i:j]
                    sub_index = list_index[i:j]
                    sublist_index.append(sub_index)
                    sublist.append(sub)
    return sublist,sublist_index

def extract_subtaglist(tag_list, full_offset, sub_offset):
    return_list = []
    last_match = 0
    for s in sub_offset:
        for i, subsuboff in enumerate(full_offset):
            if i >= last_match:
                if subsuboff[1] >= s[0]  and subsuboff[1]  <= s[1] and \
                subsuboff[2] >= s[0]  and subsuboff[2]  <= s[1] :
                    return_list.append(tag_list[i])
                    last_match = i+1
                    break
    return return_list

'''deal with the 's issue.'''
def align(tags, full_offset):
    new_offsets = []
    last_index = 0
    for i, offset in enumerate(full_offset):
        if offset[0]==tags[last_index][0]:
            new_offsets.append(offset)
            last_index += 1
        else:
            last_left = len(offset[0])
            while last_left > 0:
                start_p = offset[1] + len(offset[0])-last_left
                end_p = start_p+len(tags[last_index][0])-1
                new_offsets.append(tuple([tags[last_index][0],start_p,end_p]))
                last_left = last_left-len(tags[last_index][0])
                last_index += 1
    return new_offsets


def gen_sentence_raw(terms_arr,index):
    size = len(terms_arr)
    i = index
    skip = 0
    while (i < size):
        if (terms_arr[i][TAG_POS] in noun_tags_tweet):
            skip += 1
            i += 1
        else:
            break
    j = i-1
    while j > index:
        if (terms_arr[j][TAG_POS] in not_end):
            skip -= 1
            j -= 1
        else:
            break
    return skip

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
                print('errors in vocab list')
                print(line)
                print(len(word2idx))
                word2idx['jiyougguexjcgsnoword'] = int(line[0])
            else:
                if line[0] in word2idx.keys():
                    print('appear before:',line[0])
                    word2idx[line[0]+str(len(word2idx))] = int(line[1])
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
#F1_test(thres,model_ID,osmembed,osm_word_emb,hc,hidden,region,lstm_dim,epoch,filter_l,bool_remove,tweet_cache,osm_names, bool_pos=0, emb=1, model_type=2):
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--thres', type=float, default=0.70)
    parser.add_argument('--model_ID', type=str, default= '0319140518')
    parser.add_argument('--osmembed', type=int, default= 0)
    parser.add_argument('--osm_word_emb', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=120)
    parser.add_argument('--region', type = int, default=1)
    parser.add_argument('--lstm_dim', type = int, default=120)
    parser.add_argument('--epoch', type = int, default=11)
    parser.add_argument('--filter_l', type = int, default=1)
    parser.add_argument('--bool_remove', type = int, default=1)
    parser.add_argument('--emb', type = int, default=1)
    parser.add_argument('--model_type', type = int, default=2)

    args = parser.parse_args()
    print ('thres: '+str(args.thres))
    print ('model_ID: '+args.model_ID)
    print ('osmembed: '+str(args.osmembed))
    print ('osm_word_emb: '+str(args.osm_word_emb))
    print ('hidden: '+str(args.hidden))
    print ('lstm_dim: '+str(args.lstm_dim))
    print ('epoch: '+str(args.epoch))
    print ('filter_l: '+str(args.filter_l))
    print ('bool_remove: '+str(args.bool_remove))
    print ('emb: '+str(args.emb))
    print ('model_type: '+str(args.model_type))

    postive_pro_t =args.thres
    PAD_idx = 0
    s_max_len = 20
    bool_mb_gaze = args.osm_word_emb
    gazetteer_emb_file = 'data/osm_vector'+str(args.osmembed)+'.txt'
    bigram_file = 'model/'+args.model_ID+'-bigram.txt'
    hcfeat_file = 'model/'+args.model_ID+'-hcfeat.txt'
    START_WORD = 'start_string_taghu'
    bigram_model = load_bigram_model(bigram_file)
    if bool_mb_gaze:
        gazetteer_emb,gaz_emb_dim = load_embeding(gazetteer_emb_file)
    else:
        gazetteer_emb = []
        gaz_emb_dim = 0
    char_hc_emb,_ = load_embeding(hcfeat_file)
    word_idx_file = 'model/'+args.model_ID+'-vocab.txt'    
    word2idx, max_char_len = load_word_index(word_idx_file)
    max_char_len = 20
    
    if args.emb==2:
        glove_emb_file = 'data/GoogleNews-vectors-negative300.bin'
        emb_model = KeyedVectors.load_word2vec_format(glove_emb_file, binary=True)
        emb_dim = len(emb_model.wv['the'])
        glove_emb =  emb_model.wv
    if args.emb==3:
        glove_emb_file = 'data/glove.6B.100d.txt'
        glove_emb, emb_dim = load_embeding(glove_emb_file)
    elif args.emb==4:
        BertEmbed = BertEmbeds('data/uncased_vocab.txt', 'data/uncased_bert_vectors.txt')
        glove_emb, emb_dim = BertEmbed.load_bert_embedding()
    else:
        glove_emb_file = 'data/glove.6B.50d.txt'
        glove_emb, emb_dim = load_embeding(glove_emb_file)

    weight_l = emb_dim+gaz_emb_dim+6
    weights_matrix = np.zeros((len(word2idx.keys()), weight_l))
    weights_matrix= torch.from_numpy(weights_matrix)
    tag_to_ix = {"p": 0, "n": 1}
    HIDDEN_DIM = args.hidden
    model_path = 'model/'+args.model_ID+'epoch'+str(args.epoch)+'.pkl'
    DROPOUT = 0.5
    flex_feat_len = 3

    if args.model_type == 1:
        model = BiLSTM(weights_matrix, len(tag_to_ix), HIDDEN_DIM, 1,flex_feat_len)
        model.load_state_dict(torch.load(model_path,map_location='cpu'))
    elif args.model_type == 2:
        model = C_LSTM(weights_matrix, HIDDEN_DIM, args.filter_l, args.lstm_dim, len(tag_to_ix), flex_feat_len, DROPOUT)
        model.load_state_dict(torch.load(model_path,map_location='cpu'))
    else:
        FILTER_SIZES = [1,2,3]
        OUTPUT_DIM = 2
        model = CNN(weights_matrix, HIDDEN_DIM, FILTER_SIZES, OUTPUT_DIM, flex_feat_len, DROPOUT)
        model.load_state_dict(torch.load(model_path,map_location='cpu'))

    model.eval()
    if args.model_type ==1:
        np_word_embeds = model.word_embeds.weight.detach().numpy()
    else:
        np_word_embeds = model.embedding.weight.detach().numpy() 
    index_t = 0
    time_str = datetime.now().strftime('%m%d%H%M%S')
    raw_result_file = 'experiments/cnn_result_F1'+time_str+'m'+args.model_ID+'region'+str(args.region)+'epoch'+str(args.epoch)+'th'+str(args.thres)+'.txt'
    save_file = open(raw_result_file,'w') 
    save_file.write(model_path)
    save_file.write('\n')

    true_count = 0
    TP_count = 0
    FP_count = 0
    FN_count = 0
    tweet_cache = {}
    
    if args.region==1:
        t_json_file = "data/houston_floods_2016_annotations.json"#"data/raw_tweet.txt"
    elif args.region==2:
        t_json_file = "data/chennai_floods_2015_annotations.json"#"
    else:
        t_json_file = "data/louisiana_floods_2016_annotations.json"#"
    place_lens = {}
    detected_score = {}
    '''preload data to cache'''
    with open(t_json_file) as json_file:
        js_data = json.load(json_file)
        for key in js_data.keys():
            tweet = js_data[key]['text']
            place_names = []
            place_offset = []
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
            sentences, offsets,full_offset,hashtag_offsets = extract_sim(tweet,word2idx.keys(),1)
            sentences_lowcases = [[x.lower() for x in y] for y in sentences]
            tweet_cache[key]=[place_names,place_offset,sentences,offsets,full_offset,sentences_lowcases]
        total_sen = []
        for key in tweet_cache.keys():
            sen = ''
            for sent in tweet_cache[key][4]:
                sen += sent[0] + ' '
            total_sen.append(sen)
        tag_list = runtagger_parse(total_sen)
        tag_list=[item for item in tag_list if item]
        cur_index = 0
        index = 0
        for key in tweet_cache.keys():
            tag_lists = []
            aligned_full_offset = align(tag_list[index], tweet_cache[key][4])
            for i in range(len(tweet_cache[key][2])):
                tag_lists.append(extract_subtaglist(tag_list[index], aligned_full_offset, tweet_cache[key][3][i]))
            index += 1
            tweet_cache[key].insert(len(tweet_cache[key]),tag_lists)
            cur_index+=len(tweet_cache[key][2])
                
    with open(t_json_file) as json_file:
        js_data = json.load(json_file)
        for key in js_data.keys():
            tweet = js_data[key]['text']
            place_names, place_offset, raw_sentences, offsets, full_offset, sentences, tag_lists = tweet_cache[key]

            save_file.write('#'*50)
            save_file.write('\n')
            save_file.write(key+': '+tweet+'\n')
            ps = ''
            for place in place_names:
                for w in place:
                    ps += str(w) + ' '
                ps += '\n'
            save_file.write(ps)
            pos_str = " ".join(str(item) for item in tag_lists)
            save_file.write(pos_str)
            save_file.write('\n')

            last_remove = ['area','region']
            first_remove = ['se','ne','sw','nw']
            true_count += len(place_names)
            detected_place_names = []
            detected_offsets = []
            OSM_CONF = postive_pro_t+0.05

            for idx, sentence in enumerate(sentences):
                if sentence:
                    sub_index, all_sub_lists, _ = extract_nouns_tweet(tag_lists[idx],s_max_len)
                    all_sub_lists = [[x.lower() for x in y] for y in all_sub_lists]
                    if not all_sub_lists:
                        continue
                    cur_off = offsets[idx]
                    index_t += 1
                    osm_probs = [0]*len(all_sub_lists)
                    input_emb = np.zeros((len(all_sub_lists),s_max_len,emb_dim+gaz_emb_dim+6+flex_feat_len))
                    for i, sub_sen in enumerate(all_sub_lists):
                        sub_sen = [replace_digs(word) for word in sub_sen]
                        input_emb[i] = sentence_embeding(sub_sen, np_word_embeds,word2idx,glove_emb,\
                                                  gazetteer_emb,s_max_len,emb_dim,\
                                                  gaz_emb_dim,max_char_len,bool_mb_gaze,\
                                                 PAD_idx,START_WORD,bigram_model,char_hc_emb,flex_feat_len)

                    input_emb = torch.from_numpy(input_emb).float()
                    
                    if args.model_type==1:
                        output = model.predict(input_emb)
                        _, preds_tensor = torch.max(output, 1)
                        pos_prob = output.detach().numpy()[:,1]
                    elif args.model_type == 2:
                        output = model.predict(input_emb)
                        _, preds_tensor = torch.max(output, 1)
                        pos_prob = torch.sigmoid(output).detach().numpy()
                        pos_prob = pos_prob[:,1]
                    else:
                        tem_output = model.core(input_emb)
                        pos_prob = F.softmax(tem_output,dim=1).detach().numpy()
                        pos_prob = pos_prob[:,1]

                    for i, prob in enumerate(pos_prob):
                        if osm_probs[i] > prob:
                            pos_prob[i] = osm_probs[i]
                    postives = []

                    for i, p in enumerate(pos_prob):
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
                            if all_sub_lists[i][0] in first_remove:
                                del all_sub_lists[i][0]

#
                        detected_place_names.append(tuple(all_sub_lists[i]))
                        detected_offsets.append(tuple([cur_off[sub_index[i][0]][0],cur_off[sub_index[i][-1]][1]]))
                        save_file.write(str(round(origin_pos_prob[i],3))+':'+str(all_sub_lists[i])+'\n')
            c_tp, c_fp,c_fn, place_detect_score = interset_num(detected_offsets,place_offset,detected_place_names,place_names)
            save_file.write('tp:'+str(c_tp)+' c_fp:'+str(c_fp)+' c_fn:'+str(c_fn))
            save_file.write('\n')
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
            TP_count += c_tp
            FP_count += c_fp
            FN_count += c_fn

    P = TP_count/(TP_count+FP_count) 
    R = TP_count/(TP_count+FN_count) 
    F1 = (2*P*R) / (P+R)
    save_file.write('recall:' + str(R))
    save_file.write('\n')
    save_file.write('precision:' + str(P))
    save_file.write('\n')
    save_file.write('f1:' +  str(F1))
    save_file.write('\n')
    save_file.write('TP:' + str(TP_count))
    save_file.write('\n')
    save_file.write('FP:' + str(FP_count))
    save_file.write('\n')
    save_file.write('FN:' + str(FN_count))
    save_file.write('\n')
    save_file.write('true count:' + str(true_count))
    save_file.write('\n')
    save_file.write(json.dumps(detected_score)) # use `json.loads` to do the reverse
    save_file.write(json.dumps(place_lens)) # use `json.loads` to do the reverse
    detection_rate = [detected_score[key]/place_lens[key] for key in place_lens.keys()]
    for item in detection_rate:
        save_file.write("%s\n" % item)
    save_file.close()

if __name__ == '__main__':
    main()

