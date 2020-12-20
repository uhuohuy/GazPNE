import json, re
from utility import replace_digs,hasNumbers
import numpy as np
from core import extract, extract_sim,extract_sim2
import argparse
from wordsegment import load, segment
import pdb
#def read_tweets():
#    tweets_file = "data/louisiana_floods_2016_annotations.json"#"data/raw_tweet.txt"
#    # read tweets from file to list
#    with open(tweets_file) as f:
#        tweets = f.read().splitlines()
#    return tweets

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

def inoffset_OBI(off, off_array):
    result = 0
    for i in off_array:
        if off[0] >= i[0] and off[1]<= i[1]:
            if off[0] == i[0]:
                result = 1
            else:
                result = 2
            break
    return result


def inoffset(off, off_array):
    result = False
    for i in off_array:
        if off[0] >= i[0] and off[1]<= i[1]:
            result = True
            break
    return result

def main():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--region', type = int, default=1)
    parser.add_argument('--bool_obi', type = int, default=1)
    parser.add_argument('--savefile', type = int, default=1)

    args = parser.parse_args()
    region = args.region
    true_annotated_data_file = 'data/true_annotated'+str(args.savefile)+'.txt'
    model_ID = '0801003642'
    word_idx_file = 'data/'+model_ID+'vocab.txt'
    true_annotated = open(true_annotated_data_file,'w') 
    word2idx, max_char_len = load_word_index(word_idx_file)
    region_list = []
    if region == 1:
        region_list = [0]
    if region == 2:
        region_list = [1]
    if region == 3:
        region_list = [2]
    if region == 4:
        region_list = [0,1]
    if region == 5:
        region_list = [0,2]
    if region == 6:
        region_list = [1,2]
    if region == 7:
        region_list = [1,2,3]

    for region_index in region_list:
        if region_index==1:
            t_json_file = "data/houston_floods_2016_annotations.json"#"data/raw_tweet.txt"
        elif region_index==2:
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
#                if key == '766776546027720704':
#                    pdb.set_trace()
                sentences, offsets = extract_sim2(tweet,word2idx.keys())
                if sentences:
                    print(sentences)
                for idx, sentence in enumerate(sentences):

                    if sentence:
                        cur_off = offsets[idx]
                        true_taggs = [0]*len(sentence)
                        sen_str = ' '
                        for s in sentence:
                            sen_str = sen_str + s + ' '
                        for i, off in enumerate(cur_off):
                            if args.bool_obi:
                                true_taggs[i] = inoffset_OBI(off, place_offset)
                            else:
                                true_taggs[i] = inoffset(off, place_offset)   
                        tag_str = ' '
                        for s in true_taggs:
                            tag_str = tag_str + str(s) + ' '
                        true_annotated.write(sen_str+str(tag_str)+'\n')
    
    true_annotated.close()

if __name__ == '__main__':
    main()
