import csv
import re
import copy
import codecs
import random
from random import randint
DIGIT_RE = re.compile(r"\d")
import sys
import collections
#from process_pos import get_variable_words
from utility import write_place, hasNumbers
from gensim.models import KeyedVectors
from utility import load_osm_names, isascii, split_numbers, replace_digs,load_osm_names_fre
from Garzetter_sim_pre import load_embeding
import inflect
import string
import argparse
import time
from memory_profiler import profile


engine = inflect.engine()
print(sys.getdefaultencoding())
'''For each ascill char in the lowercase, combine it with a number string ('0','00','000', '0000', and '00000')
 to form a new negative example. <count> represents  the count of repetively producing new negative 
examples each ascill char'''

def ascii_num_neg(count):
    ascii_w = list(string.ascii_lowercase)
    new_neg = []
    max_bit = 6
    for i in ascii_w:
        for j in range(max_bit):
            cur_n_s = '0'*(j+1)
            for t in range(count):
                new_test = []
                new_test.append(cur_n_s)
                new_test.append(i)
                new_neg.append(tuple(new_test))
                new_test = []
                new_test.append(i)
                new_test.append(cur_n_s)
                new_neg.append(tuple(new_test))
    return new_neg


'''randomly choose one place name from <place_names> if the last word of place name 
is in <last_word> and one word from <last_word>. Insert the word at the end or begining of the place name
 to form a new negative example. <max_count> denotes the maximum number of negative examples can be generated'''
def insert_last_word(place_names, last_word, max_count):
    new_neg = []
    insert_len = [1]
    place_len = len(place_names)
    vocab_len = len(last_word)
    for i in range(max_count):
        random_indexs_p = random.sample(range(place_len), 1)
        place = place_names[random_indexs_p[0]]
        while place[-1] not in last_word:
            random_indexs_p = random.sample(range(place_len), 1)
            place = place_names[random_indexs_p[0]]
        for le in insert_len:
            random_indexs = random.sample(range(vocab_len), le)
            new_list = list(copy.deepcopy(place))
            random_insert_p = random.sample([len(new_list),0], 1)
            for index_l in random_indexs:
                new_list.insert(random_insert_p[0], last_word[index_l])
            new_neg.append(tuple(new_list))
    return new_neg

'''randomly choose one place name from <place_names> and k (in <insert_len>) words from <vocabs>. Insert the k 
words at random position of the place name to form a new negative example. <max_count> denotes the maximum 
number of negative examples can be generated'''

def get_neg_insert_random_gen(place_names, vocab_list, max_count, insert_len = [1,2]): 
    place_len = len(place_names)
    vocab_len = len(vocab_list)
    neg_pla = []
    count = int(max_count/len(insert_len))
    for i in range(count):
        for le in insert_len:
            random_indexs_p = random.sample(range(place_len), 1)
            random_indexs = random.sample(range(vocab_len), le)
            new_list = list(copy.deepcopy(place_names[random_indexs_p[0]]))
            p_l = len(new_list)
            if p_l > 1:
                indexs_insert = random.sample(range(1,p_l), 1)
                for index_l in random_indexs:
                    new_list.insert(indexs_insert[0],vocab_list[index_l])
                neg_pla.append(tuple(new_list))
    return neg_pla

'''randomly choose two place names from <place_names> and k (in <insert_len>) words from <vocabs>. Insert the k 
words between the two place names to form a new negative example. <max_count> denotes the maximum number of 
negative examples can be generated'''


def combine_neg_gen(place_names, vocabs, max_count, insert_len = [1]):
    place_len = len(place_names)
    vocab_len = len(vocabs)
    neg_pla = []
    com_num = 2
    count = int(max_count/len(insert_len))
    for i in range(count):
        for le in insert_len:
            random_indexs_p = random.sample(range(place_len), com_num)
            random_indexs = random.sample(range(vocab_len), le)
            new_list = []
            new_list.extend(list(place_names[random_indexs_p[0]]))
            for index_l in random_indexs:
                new_list.append(vocabs[index_l])
            new_list.extend(list(place_names[random_indexs_p[1]]))
            neg_pla.append(tuple(new_list))
    return neg_pla

def combine_multi(place_names, vocabs, most_gen_vocabs, max_count, insert_len = [1,2]):
    place_len = len(place_names)
    vocab_len = len(vocabs)
    most_gen_len = len(most_gen_vocabs)
    neg_pla = []
    count = int(max_count/len(insert_len))
    for i in range(count):
        for le in insert_len:
            random_indexs_p = random.sample(range(place_len), 1)
            random_indexs = random.sample(range(vocab_len), le)
            random_indexs_m = random.sample(range(most_gen_len), 1)
            new_list = []
            for index_l in random_indexs:
                new_list.append(vocabs[index_l])
            new_list.append(most_gen_vocabs[random_indexs_m[0]])
            new_list.extend(list(place_names[random_indexs_p[0]]))
            neg_pla.append(tuple(new_list))
    return neg_pla


'''randomly choose 2 place names from <place_names> and combine them as a new negative example. <max_count> denotes 
the maximum number of negative examples can be generated'''

def combine_neg(place_names, max_count):
    place_len = len(place_names)
    neg_pla = []
    com_num = 2
    for i in range(max_count):
        random_indexs = random.sample(range(place_len), com_num)
        temp_p = []
        for p in random_indexs:
            temp_p.extend(list(place_names[p]))
        neg_pla.append(tuple(temp_p))
    return neg_pla



''' for each word in <last_word> named a and each word in <most_general> named b, combine a and b to 
form a new negative example. The frequency (in <last_word_fre>) of a word in <last_word> determines the 
count of repetively producing new negative examples from the word. <max_count> denotes the maximum number 
of examples can be generated'''

def general_last_word_adv(last_word, last_word_fre, most_general, max_count, gene_count = 3):
    return_places = []
    for word in last_word:
        for g in range(1, gene_count):
            count = int(max_count*last_word_fre[word]/(gene_count-1))
            for i in range(count):
                random_indexs = random.sample(range(len(most_general)), g+1)
                gen_words = [most_general[j] for j in random_indexs]
                gen_words.append(word)
                return_places.append(tuple(gen_words))
    return return_places

''' for each word in <last_word> named a and each word in <most_general> named b, combine a and b to 
form a new negative example. <count> represents the time of copying a new negative example'''

def general_last_word2(last_word, most_general, count):
    return_places = []
    for word in last_word:
        for gen in most_general:
            for i in range(count):
                gen_words = [gen, word]
                return_places.append(tuple(gen_words))
    return return_places

def general_first_word2(first_word, most_general, count):
    return_places = []
    for word in first_word:
        for gen in most_general:
            for i in range(count):
                gen_words = [word, gen]
                return_places.append(tuple(gen_words))
    return return_places

def general_first_word_adv(first_word, first_word_fre, most_general, max_count, gene_count = 3):
    return_places = []
    for word in first_word:
        for g in range(1, gene_count):
            count = int(max_count*first_word_fre[word]/(gene_count-1))
            for i in range(count):
                random_indexs = random.sample(range(len(most_general)), g+1)
                gen_words = [most_general[j] for j in random_indexs]
                gen_words.insert(0, word)
                return_places.append(tuple(gen_words))
    return return_places


def load_f_data(pos_f, very_fre_count):
    pos_training_data = {}
    count = 0
    very_fre_words = []
    with codecs.open(pos_f, 'r',encoding='utf-8') as file:
        for line in file:
            count += 1
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = re.split("\s",line)
            tokens = list(filter(None, tokens))
            tokens = [x for x in tokens if x]
            pos_training_data[tokens[1].lower()] = int(tokens[3])
            if len(very_fre_words) < very_fre_count:
                very_fre_words.append(tokens[1])
    return pos_training_data,very_fre_words

'''randomly choose k (in <place_lens>) from <words>  and combine them as a new negative example. <max_count> denotes 
the maximum number of examples can be generated'''

def general_words_neg_dig(words, max_count, place_lens = [2,3,4]):
    neg_names = []
    ave_count = int(float(max_count) / len(place_lens))
    for l in place_lens:
        for i in range(ave_count):
            random_indexs = random.sample(range(1, len(words)-1), l)
            neg_names.append(tuple([words[idx] for idx in random_indexs]))
    return neg_names

'''randomly choose k (in <place_lens>) words from <words> and combine them as a new negative example if it satisifies 
several rules, invloving <last_words>, <first_words>, and <very_general_words>. <max_count> denotes 
the maximum number of examples can be generated'''
def general_words_neg(words, max_count, last_words, first_words, very_general_words, place_lens = [2,3,4,5,6]):
    neg_names = []
    ave_count = int(float(max_count) / len(place_lens))
    for l in place_lens:
        for i in range(ave_count):
            random_indexs = random.sample(range(1, len(words)-1), l)
            while words[random_indexs[-1]] in last_words and l <=3 and (words[random_indexs[0]] in  first_words or words[random_indexs[0]] not in very_general_words):
                random_indexs = random.sample(range(1, len(words)-1), l)
            neg_names.append(tuple([words[idx] for idx in random_indexs]))
    return neg_names

'''for each name in <place_names>, get the sub set of this name if 
its last word is not in <last_word_set>'''

def short_neg(place_names, last_word_set, first_words):
    neg_vocab = []#
    sing_neg_words = []
    for place in place_names:
        neg_len = list(range(1,len(place)))
        for i in neg_len:
            for j in range(len(place)-i+1):
                temp_p = tuple(place[j:j+i])
                if  (temp_p[-1] not in last_word_set and temp_p[0] not in first_words) or len(temp_p)==1: # or (temp_p[-1] in last_word_set and temp_p[0] not in fist_word_set)
                    neg_vocab.append(temp_p)
                    if len(temp_p) == 1 and temp_p[-1] not in last_word_set:
                        sing_neg_words.append(temp_p)
    return neg_vocab,sing_neg_words

'''for each word in <negative_prex>, combine it with numbers ('0','00','000', '0000', and '00000') 
to form a new negative example. <count> represents the count of repetively producing new negative 
examples from a word in <negative_prex>'''

def negative_prefix_num(negative_prex, count, number=6):
    negative_places = []
    multi=int(count/number)+1
    for prex in negative_prex:
        for i in range(multi):
            for index_e in range(1,number):
#                index_e = randint(1, 6)
                negative_places.append(tuple(['0'*index_e+prex[0]]))
                negative_places.append(tuple([prex[0]+'0'*index_e]))
                negative_places.append(tuple(['0'*index_e,prex[0]]))
                negative_places.append(tuple([prex[0],'0'*index_e]))
                #negative_places.append(tuple([prex[0],'0'*index_e]))
                temp_p = list(prex[0])
                temp_p.append('0'*index_e)
                negative_places.append(tuple(temp_p))
                temp_p1 = ['0'*index_e]
                tt = list(prex[0])
                temp_p1.extend(tt)
                negative_places.append(tuple(temp_p1))
    return negative_places



'''randomly choose at most <pair> paris of  words from <vocab_list> and insert them in the begining and end 
of a positive name from <place_names> respectively. <count> represents 
 the count of repetively producing new negative examples from a place name in <place_names>'''

def get_neg_insert(place_names, vocab_list, before_num_words, last_word_set, first_word_set, count=1, pair = 2):
    new_neg = []
    vocab_len = len(vocab_list)
    for place in place_names:
        for c in range(count):
            for i in range(0,pair):
                new_list = list(copy.deepcopy(place))
                for j in range(0,i+1):
                    index_s = randint(0, vocab_len-1)
                    index_e = randint(0, vocab_len-1)
                    while ((j==i) and (tuple([new_list[-1]]) in before_num_words) and (vocab_list[index_e].isdigit())) or (i==0 and vocab_list[index_e] in last_word_set and vocab_list[index_s] in first_word_set):
                        index_e = randint(0, vocab_len-1)
                    new_list.insert(0, vocab_list[index_s])
                    new_list.append(vocab_list[index_e])
                new_list = tuple(new_list)
                new_neg.append(new_list)
    return new_neg

'''randomly choose k (in <insert_len>) words from <vocab_list> and insert them in the end 
of a positive name from <place_names> if the last word of the positive name is in <last_word_set>. <count> represents 
 the count of repetively producing new negative examples from a place name in <place_names>'''


def get_neg_insert_last(place_names, vocab_list, last_word_set, count=5, insert_len = [1,2,3]):
    new_neg = []
    vocab_len = len(vocab_list)
    for place in place_names:
        for c in range(count):
            for le in insert_len:
                random_indexs = random.sample(range(vocab_len), le)
                new_list = list(copy.deepcopy(place))
                if not (le==1 and vocab_list[random_indexs[0]] in last_word_set):
                    for index_l in random_indexs:
                        new_list.append(vocab_list[index_l])
                    new_neg.append(tuple(new_list))
    return new_neg

#def get_neg_insert_last(place_names, vocab_list, last_word_set, count=5, insert_len = [1,2,3]):
#    new_neg = []
#
#    vocab_len = len(vocab_list)
#    for place in place_names:
##        if place[-1] in last_word_set:
#        for c in range(count):
#            for le in insert_len:
#                random_indexs = random.sample(range(vocab_len), le)
#                new_list = list(copy.deepcopy(place))
#                for index_l in random_indexs:
#                    new_list.append(vocab_list[index_l])
#                new_neg.append(tuple(new_list))
#    return new_neg

#insert a couple of general words in the beginning of a place entities to generate negative 
# examples if the inersted word is not in the preposition_list ['at','in','on','to']
    

'''randomly choose k (in <insert_len>) words from <vocab_list> and insert them in the beginning 
of a positive name from <place_names> . <count> represents 
 the count of repetively producing new negative examples from a place name in <place_names>'''


def get_neg_insert_first(place_names, vocab_list, preposition_list, count=10, insert_len = [1,2,3]):
    new_neg = []
    vocab_len = len(vocab_list)
    for place in place_names:
        for c in range(count):
            for le in insert_len:
                random_indexs = random.sample(range(vocab_len), le)
                while le==1 and vocab_list[random_indexs[0]] in preposition_list:
                    random_indexs = random.sample(range(vocab_len), le)                    
                new_list = list(copy.deepcopy(place))
                for index_l in random_indexs:
                    new_list.insert(0,vocab_list[index_l])
                new_neg.append(tuple(new_list))
    return new_neg


#def get_neg_insert_last_num(place_names, vocab_list, last_num_word_set, max_count):
#    new_neg = []
#    insert_len = [1,2]
#    vocab_len = len(vocab_list)
#    count = 0
#    for place in place_names:
#        if count < max_count and place[-1] not in last_num_word_set:
#            for le in insert_len:
#                random_indexs = random.sample(range(vocab_len), le)
#                new_list = list(copy.deepcopy(place))
#                for index_l in random_indexs:
#                    new_list.append(vocab_list[index_l])
#                new_neg.append(tuple(new_list))
#                count += 1
#    return new_neg

'''randomly choose k (in <insert_len>) words from <vocab_list> and insert them in the end 
of a positive name from <place_names> if the last word of the positive 
name is not in <last_alpha_word_set>. <max_count> denotes the maximum number 
of examples can be generated'''

def get_neg_insert_alpha(place_names, vocab_list, last_alpha_word_set, max_count, insert_len = [1,2]):
    new_neg = []
    vocab_len = len(vocab_list)
    count = 0
    for place in place_names:
        if count<max_count and place[-1] not in last_alpha_word_set:
            for le in insert_len:
                random_indexs = random.sample(range(vocab_len), le)
                new_list = list(copy.deepcopy(place))
                for index_l in random_indexs:
                    new_list.append(vocab_list[index_l])
                new_neg.append(tuple(new_list))
                count += 1
    return new_neg


'''randomly choose k (in <insert_len>) words from <vocab_list> and insert them in the end 
of a positive name from <place_names> if the last word of the positive 
name is not in <last_alpha_word_set>. <max_count> denotes the maximum number 
of examples can be generated'''

def get_neg_insert_alpha_gen(place_names, vocab_list, last_alpha_word_set, max_count, insert_len = [1,2]):
    new_neg = []
    vocab_len = len(vocab_list)
    count = 0
    pl_len = len(place_names)
    while count<max_count:
        for le in insert_len:
            random_indexs = random.sample(range(vocab_len), le)
            random_indexs_p = random.sample(range(pl_len), 1)
            place = place_names[random_indexs_p[0]]
            while place[-1] in last_alpha_word_set:
                random_indexs_p = random.sample(range(pl_len), 1)
                place = place_names[random_indexs_p[0]]
            new_list = list(copy.deepcopy(place))
            for index_l in random_indexs:
                new_list.append(vocab_list[index_l])
            new_neg.append(tuple(new_list))
            count += 1
    return new_neg

'''randomly choose k (in <insert_len>) words from <vocab_list> and insert them in the begining 
of a positive name from <place_names> if the first word of the positive 
name is not in <first_num_word_set>. <max_count> denotes the maximum number 
of examples can be generated'''

def get_neg_insert_first_num(place_names, vocab_list, first_num_word_set, max_count, insert_len = [1,2]):
    new_neg = []
    vocab_len = len(vocab_list)
    count = 0
    for place in place_names:
        if count < max_count and place[0] not in first_num_word_set:
            for le in insert_len:
                random_indexs = random.sample(range(vocab_len), le)
                new_list = list(copy.deepcopy(place))
                for index_l in random_indexs:
                    new_list.insert(0,vocab_list[index_l])
                new_neg.append(tuple(new_list))
                count += 1
    return new_neg

'''randomly choose a word from <prex_num> and insert it in the begining 
and end of a positive name from <place_names>.
 <max_count> denotes the maximum number of examples can be generated'''
def get_neg_insert_prex_num(place_names, prex_num, max_count):
    new_neg = []
    vocab_len = len(prex_num)
    count = 0
    for place in place_names:
        if count <  max_count:
            # insert prex num at the end
            index_l = randint(0, vocab_len-1)
            new_list = list(copy.deepcopy(place))
            new_list.extend(list(prex_num[index_l]))
            new_neg.append(tuple(new_list))
            count += 1

            # insert prex num at the begining
            index_l = randint(0, vocab_len-1)
            new_list = list(copy.deepcopy(place))
            temp = list(prex_num[index_l])
            temp.extend(new_list)
            new_neg.append(tuple(temp))
            count += 1
    return new_neg

'''randomly choose 1 to <insert_word_len> words from <vocab> and insert them 
in the beginning and end of a place name from <place_names>. <count> represents 
the count of repetively producing a new negative example from a place name in <place_names>'''
def get_neg_insert_prex_gen(place_names, vocab, count=3, insert_word_len=2):
    new_neg = []
    vocab_len = len(vocab)
    for place in place_names:
        for c in range(count):
            for i in range(0,insert_word_len):
                new_list = []
                for j in range(0,i+1):
                    index_s = randint(0, vocab_len-1)
                    new_list.append(vocab[index_s])
                new_list.extend(list(place))
                new_list = tuple(new_list)
                new_neg.append(new_list)
                
    for place in place_names:
        for c in range(count):
            for i in range(0,insert_word_len):
                new_list = list(place)
                for j in range(0,i+1):
                    index_s = randint(0, vocab_len-1)
                    #while not vocab_list[index_s].isalnum():
                    #    index_s = randint(0, vocab_len-1)
                    new_list.append(vocab[index_s])
                new_list = tuple(new_list)
                new_neg.append(new_list)
                
    return new_neg

'''adjust the order of words of a true place name to form a negative place name'''
def adjust_order_neg(place_names, last_words):
    new_places = []
    gen_places_c = 6
    dis_match_thres = 4
    for place in place_names:
        p_l = len(place)
        if p_l >= 4:
            ori_order = list(range(0,p_l))
            found_count = 0
            while found_count < gen_places_c:
                new_order = random.sample(ori_order,p_l)
                found_count += 1
                if len ([i for i, j in zip(ori_order, new_order) if i != j]) >= dis_match_thres:
                    if new_order[0] != ori_order[0] and new_order[-1] != ori_order[-1] and place[new_order[-1]] not in last_words:
                        temp_place = tuple([place[pi] for pi in new_order ])
                        #if temp_place not in place_names:
                        new_places.append(temp_place)
    return new_places

'''randomly choose 1 to <insert_word_len> words and insert them at the random position of a place name from <place_names>'''
def get_neg_insert_random(place_names, vocab_list, insert_word_len = 4):
    new_neg = []
    vocab_len = len(vocab_list)
    for place in place_names:
        for i in range(0,insert_word_len):
            new_list = []
            new_list.extend(list(place))
            for j in range(0,i+1):
                index_s = randint(0, vocab_len-1)
                #while not vocab_list[index_s].isalnum():
                #    index_s = randint(0, vocab_len-1)
                insert_p = randint(0, len(new_list))
                new_list.insert(insert_p, vocab_list[index_s])
            new_list = tuple(new_list)
            new_neg.append(new_list)
    return new_neg


'''get the index of the second string list in the first string list'''

def sub_str(first, second):
    count = 0
    for i in range(len(second)-len(first)+1):
        count = 0
        for j in range(len(first)):
             if second[i+j] == first[j]:
                 count += 1
        if count == len(first):
            return (i,i+len(first)-1)
    if count == len(first):
        return (i,i+len(first)-1)
    else:
        return []

'''read the abr_file and load the <word, abbreviation> pairs'''

def abbrevison(abr_file):
    with open(abr_file,mode='r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        abbr = {}
        for row in reader:
            tokens = row[0].split(' ')
            if len(tokens) == 1:
                abbr[row[1]] = tokens[0] 
    return abbr

'''extend the positive examples by replace the word in the original place name with the 
   corresponding abbreviation word in the abr_file'''

def replace_abbrevison(place_names,abr_file):
    #get abbrevision of place words
    aug_place_names = []
    with open(abr_file,mode='r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        full = [];
        abbr = []
        for row in reader:
            tokens = row[0].split(' ')
            full.append(tokens)
            abbr.append(row[1])
            for place in place_names:
                index = sub_str(tokens, place)
                if index:
                    new_list = []
                    bool_f = False
                    for i, p in enumerate(place):
                        if i not in index:    
                            new_list.append(p)
                        else:
                            if not bool_f:
                                new_list.append(row[1])
                                bool_f = True
                    aug_place_names.append(tuple(new_list))
    #import pdb
    #pdb.set_trace()
    place_names = [tuple(p) for p in place_names]
    place_names.extend(aug_place_names)
    return place_names, full,abbr


'''copy an element mulitple times according to the value of the element in the dictionary'''
def copyACount(neg, place_dic, fre_thres = 1):
    new_neg = []
    for pla in neg:
        if place_dic[pla] >= fre_thres:
            for i in range(place_dic[pla]):
                new_neg.append(pla)
    return new_neg

def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--file', type=int, default=29)
    parser.add_argument('--ht', type=int, default=200)
    parser.add_argument('--lt', type=int, default=150)
    parser.add_argument('--ft', type=int, default=10)
    parser.add_argument('--bool_prepos', type=int, default=0)
    parser.add_argument('--unseen', type=int, default=0)
    parser.add_argument('--bool_insert_first', type=int, default=0)#
    parser.add_argument('--bool_rm_pla_f', type=int, default=0)
    parser.add_argument('--multiple_f_l', type=int, default=1)
    parser.add_argument('--geonames', type=int, default=1)
    parser.add_argument('--general_word', type=int, default=26000)
    parser.add_argument('--bool_disorder', type=int, default=1)
    parser.add_argument('--bool_short', type=int, default=1)


    args = parser.parse_args()
    print ('bool_prepos: '+str(args.bool_prepos))
    print ('unseen: '+str(args.unseen))
    print ('file: '+str(args.file))
    print ('ht: '+str(args.ht))
    print ('lt: '+str(args.lt))
    print ('ft: '+str(args.ft))
    print ('bool_insert_first: '+str(args.bool_insert_first))
    print ('bool_rm_pla_f: '+str(args.bool_rm_pla_f))
    print ('multiple_f_l: '+str(args.multiple_f_l))
    print ('geonames: '+str(args.geonames))
    print ('general_word: '+str(args.general_word))
    print ('bool_disorder: '+str(args.bool_disorder))
    print ('bool_short: '+str(args.bool_short))


    unseen_words = ['hiagnnamalnsw']

    file_name = 'data/osm_abbreviations.csv'
    bool_highway_check = True
    bool_single_name_aug = True
    bool_comb_neg_gen = True
    bool_alpha_gen = True
        
    sim_abv = abbrevison(file_name)
    argu_count = 150
    highway_mark = {}
    osm_file = 'data/us.tsv'
    chennai_osm_file = 'data/india.tsv'
    place_l = {}
    target_files = []
    target_files.append(chennai_osm_file)
    target_files.append(osm_file)
    for raw_file in target_files:
        with open(raw_file,encoding='utf-8') as tsvfile:
            reader = csv.DictReader(tsvfile, dialect='excel-tab')
            for row in reader:
    #            if not isascii(row['name']):
    #                print(row['name'])
                t_row_nobrackets = re.sub("[\(\[].:;*?[\)\]]", "", row['name'])
                t_corpus = [word.lower() for word in re.split("[. #,&\"\',’]",t_row_nobrackets)]
                if len(t_corpus) in place_l.keys():
                    place_l[len(t_corpus)] += 1
                else:
                    place_l[len(t_corpus)] = 1
    #
                if len(t_corpus)==1 and (replace_digs(t_corpus[0])== 'street' or replace_digs(t_corpus[0])== 'etta'): # or replace_digs(t_corpus[0])== 'mississippi'
                    print(row)
#                   pdb.set_trace()
                if row['class'] == 'highway' and (row['type'] == 'residential' or row['type'] == 'primary'  \
                      or row['type'] == 'secondary'  or row['type'] == 'trunk' \
                      or row['type'] == 'tertiary' or row['type'] == 'unclassified'):
                    row_nobrackets = re.sub("[\(\[].:;*?[\)\]]", "", row['name'])         
                    corpus = [word.lower() for word in re.split("[. #,&\"\',’]",row_nobrackets)]
                    if corpus[-1] in sim_abv.keys():
                        corpus[-1] = sim_abv[corpus[-1]]
                    if corpus[-1] in highway_mark.keys():
                        highway_mark[corpus[-1]] =  highway_mark[corpus[-1]] +1
                    else:
                        highway_mark[corpus[-1]] = 0
#    return
    high_way_thres = args.ht
    highway_end = []
    highway_value = []
    for c in highway_mark.keys():
        if highway_mark[c] >  high_way_thres:
            highway_end.append(c)
            highway_value.extend([c]*highway_mark[c])
#            print(c+':' +str(highway_mark[c]))
    neg_file = 'data/negative'+str(args.file)+'.txt'
    full_neg_file = 'data/full_negative'+str(args.file)+'.txt'

    pos_file = 'data/positive'+str(args.file)+'.txt'
    addition_file = 'data/addition.txt'

    # extract general words from embeddings
    glove_emb_file = 'data/glove.6B.50d.txt'
    glove_emb, emb_dim = load_embeding(glove_emb_file)
    google_emb_file = 'data/GoogleNews-vectors-negative300.bin'
    emb_model = KeyedVectors.load_word2vec_format(google_emb_file, binary=True)
    google_emb = emb_model.wv
    general_words_excep = ['monday','tuesday','wednesday','thursday','friday','saturday','sunday']
    #google_word_index = 8000
    max_count = 1000000000
    
    choosen_dig_count = 10;
    most_general_count = 240
    most_general_count_2 = 150
    most_general_words = []
    more_general_count = 1000
    more_general_count_2 = 4800
    more_general_words = []
    w2c = dict()
    for item in google_emb.vocab:
        w2c[item]=google_emb.vocab[item].count
    w2cSorted=dict(sorted(w2c.items(), key=lambda x: x[1],reverse=True))   
    #for (k, v) in w2cSorted:
     #   pdb.set_trace()
     #   if v == len(w2cSorted)-google_word_index:
      #      print(w2c['k'])
    google_vocabs = []
    g_l = len(google_emb.vocab)
    general_word_t = args.general_word
    general_multi = 100
    temp_ignore = set([])
    very_gen_words = []
    rem_sym_negs = []
    number_words = []
    glove_num_prefix = []
    glove_num_prefix_set = set([])

    for vocab in google_emb.vocab:
        if isascii(vocab) and vocab.islower():
            bool_general = False
            if hasNumbers(vocab):
                number_words.append(vocab)
                front, back = split_numbers(vocab)
                if front:
                    glove_num_prefix.append(tuple([front.lower()]))
                    glove_num_prefix.append(tuple([engine.plural(front.lower())]))
                    glove_num_prefix_set.add(front.lower())
                    glove_num_prefix_set.add(engine.plural(front.lower()))

                if back:
                    glove_num_prefix.append(tuple([back.lower()]))
                    glove_num_prefix.append(tuple([engine.plural(back.lower())]))
                    glove_num_prefix_set.add(back.lower())
                    glove_num_prefix_set.add(engine.plural(back.lower()))
#            if 'houston' == vocab.lower():
#                pdb.set_trace()
            if w2cSorted[vocab] >  g_l - general_word_t:
                if vocab.lower() not in temp_ignore:
                    very_gen_words.append(vocab.lower())
                    bool_general = True
            else:
                temp_ignore.add(vocab.lower())
                c_count = very_gen_words.count(vocab.lower())
                for cc in range(c_count):
                    very_gen_words.remove(vocab.lower())
            if w2cSorted[vocab] >  g_l - most_general_count:
                if vocab.lower().isalpha():
                    most_general_words.append(vocab.lower())
                else:
                    most_general_count = most_general_count + 1
            if w2cSorted[vocab] >  g_l - more_general_count:
                if vocab.lower().isalpha():
                    more_general_words.append(vocab.lower())
                else:
                    more_general_count = more_general_count + 1
                    
            temp = re.split('[^a-zA-Z^0-9]',vocab.lower())
            temp_sym = []
            for x in temp:
                groups = re.split('(\d+)',x)
                temp_sym.extend(groups)
            temp_sin = [x for x in temp_sym if x]
            if "'" in vocab.lower():
                if bool_general:
                    multi_count = general_multi
                else:
                    multi_count = 1
                rem_sym_negs.extend([tuple(temp_sin)]*multi_count)
            for w in temp_sin:
                google_vocabs.append(tuple([replace_digs(w.lower())]))
                #rem_sym_negs.append(temp_sym)
    # extract single word place name from osm
    glove_vocabs = []
    glove_nums = []
    for w in glove_emb.keys():
        if isascii(w):
            if hasNumbers(w):
                number_words.append(w)
                front, back = split_numbers(w)
                if front:
                    glove_num_prefix.append(tuple([front.lower()]))
                    glove_num_prefix.append(tuple([engine.plural(front.lower())]))
                    glove_num_prefix_set.add(front.lower())
                    glove_num_prefix_set.add(engine.plural(front.lower()))
                if back:
                    glove_num_prefix.append(tuple([back.lower()]))
                    glove_num_prefix.append(tuple([engine.plural(back.lower())]))
                    glove_num_prefix_set.add(back.lower())
                    glove_num_prefix_set.add(engine.plural(back.lower()))
            #glove_vocabs.append(tuple([replace_digs(w.lower())]))
            temp = re.split('[^a-zA-Z^0-9]',w.lower())
            temp_sym = []
            for x in temp:
                groups = re.split('(\d+)',x)
                temp_sym.extend(groups)
            temp_sin = [x for x in temp_sym if x]
            for sw in temp_sin:
                glove_vocabs.append(tuple([replace_digs(sw.lower())]))
    write_place('num_gen.txt', number_words)
    glove_num_prefix_set = list(glove_num_prefix_set)

    single_osm_pl = load_osm_names('data/single_word_place.txt')
    google_vocabs.extend(glove_vocabs)
    pure_general_vocabs = list(set(copy.deepcopy(google_vocabs)))
    print(len(pure_general_vocabs))
    google_vocabs = list(set(google_vocabs).difference(set(single_osm_pl)))
    write_place('vocabgoogle.txt',google_vocabs)



    vocab_list = []
    general_word_count = 1000000.0
    word_f_f = 'data/word_frequency.txt'

    word_fre_ori, most_gen_words_2 = load_f_data(word_f_f, most_general_count_2)
    word_fre_ori2, more_gen_words_2 = load_f_data(word_f_f, more_general_count_2)
    most_general_words.extend(most_gen_words_2)
    most_general_words = list(set(most_general_words))
    more_general_words.extend(more_gen_words_2)
    more_general_words = list(set(more_general_words))
    total_w_fres = sum(list(word_fre_ori.values()))
#    extend_words = get_variable_words()
    new_general_word = []
    word_fre = []
    for word in word_fre_ori.keys():
        word_fre.append(word)
        new_general_word.extend([tuple([word.lower()])]*(int(word_fre_ori[word]*general_word_count/total_w_fres)))
#        for var_words in extend_words:
#            if word in var_words:
#                word_fre.extend(var_words)
    general_words_first = []
    #general_words_first.extend(word_fre)
    general_words_first.extend(most_general_words)
    general_words_first = list(set(general_words_first))
#    general_words_first.append('of')
#    general_words_first.append('flooded')
    word_fre.extend(very_gen_words)
    word_fre.extend(general_words_excep)
    word_fre.extend(list(string.ascii_lowercase))
    word_fre.append('0')
    word_fre.append('00')
    word_fre.append('000')
    word_fre.append('0000')
    word_fre.append('00000')
    word_fre.append('000000')
    word_fre.append('0000000')
    word_fre.append('00000000')
    word_fre.append('000000000')
    word_fre.append('0000000000')
    word_fre.append('00000000000')
    word_fre.append('000000000000')

    word_fre = list(set(word_fre))
    google_vocabs.extend(new_general_word)
    write_place('test.txt', google_vocabs)
    google_vocabs = [w[0] for w in google_vocabs]
    place_names = []
    single_place_names =  []
    count = 0
    cc = 0
    osm_result_file = 'data/osm_raw_new.txt'
    osm_result = open(osm_result_file,'w')
    #get the last word of a highway objectbool_comb_multi2
    last_num_count = 0
    ofv_count = 0
    high_way_augmented = []
    postive_add = []
    state_augmented = []
    single_places = set()
    '''extract raw place names from OSM'''
    with open(osm_file,encoding='utf-8') as tsvfile, open(chennai_osm_file,encoding='utf-8') as chennai_svfile:
        reader_base = csv.DictReader(tsvfile, dialect='excel-tab')
        reader_chennai = csv.DictReader(chennai_svfile, dialect='excel-tab')
        readers = []
        readers.append(reader_chennai)
        readers.append(reader_base)
        for reader in readers:
            for row in reader:
    #            if row['name'].lower() == 'houston':
    #                pdb.set_trace()
                if  isascii(row['name']) and count < max_count:
                    osm_result.write(row['name'])
                    osm_result.write('\n')
                    ignore = True
                    if 'footway' not in row['type'] and 'service' not in row['type'] and 'cycleway' not in row['type']  \
                            and 'track' not in row['type']  and 'path' not in row['type'] and 'locality' not in row['type']: #and 'landuse' not in row['class']
                        ignore = False
                    if (row['class'] == 'highway' and (row['type'] == 'residential' or row['type'] == 'primary'  \
                          or row['type'] == 'secondary'  or row['type'] == 'trunk' \
                          or row['type'] == 'tertiary' or row['type'] == 'unclassified')) or \
                           (row['osm_type'] == 'way' and ('secondary' in row['type'] or 'tertiary' in row['type'])):
                        is_highway = True
                    else:
                        is_highway = False
                        # or (row['class'] == 'boundary' and row['type'] == 'administrative')
                    if ((row['class'] == 'place' and (row['type'] == 'city' or row['type'] == 'town' or row['type']  == 'suburb' or \
                         row['type']  == 'county'))) : #and row['type'] != 'locality'
                        is_arg = True
                        # enhance the place entity in the type of state since it is important but less mentioned on OSM.
                        # By doing so, 'louissanan' is extended to 'louissana state'
                        if  row['state']:
                            row_nobrackets = re.sub("[\(\[].:;*?[\)\]]", "", row['state'])         
                            sta_corpus = [word.lower() for word in re.split("[. #,&\"\',’]",row_nobrackets)]
                            if sta_corpus[-1] == 'state':
                                state_augmented.append(tuple(sta_corpus))
                            else:
                                sta_corpus.append('state')
                                state_augmented.append(tuple(sta_corpus))
                    else:
                        is_arg = False
                    
                    if not ignore:
                        row_nobrackets = re.sub("[\(\[].:;*?[\)\]]", "", row['name'])         
                        corpus = [word.lower() for word in re.split("[. #,&\"\',’]",row_nobrackets)]
                        #if corpus == ['duplin', 'county']:
                        #    pdb.set_trace()
                        new_corpus = []
                        for cor in corpus:
                            all_ascii = ''.join(char for char in cor if ord(char) < 127)
                            new_corpus.append(all_ascii)
                        corpus = [x for x in new_corpus if x and (len(x) < 2 or (len(x)>=2 and not (x[0]== '(' and x[len(x)-1]== ')')))]
                        # tackel the 9-2 and 9/2 issues
                        temp_corpus = []
                        bool_ignored = False
                        for w in corpus:
                            if  '-' in w or '/' in w:
        
                                if w != '-' and w != '/':
                                    bool_dig = False
                                    subs = re.split("-|/",w)
                                    if '/' in w:
                                        bool_ignored = True
                                    temp_corpus.extend(subs)
                                if w == '/':
                                    bool_ignored = True
                        
                            else:
                                temp_corpus.append(w)
    
                        if not bool_ignored:
                            corpus = temp_corpus
                        else:
                            corpus = []
                            cc += 1
                        real_corpus = []
                        for x in corpus:
                            if ';' in x:
                                temps = x.split(';')                    
                                for i in range(len(temps)):
                                    if i < len(temps)-1:
                                        if temps[i]:
                                            real_corpus.append(temps[i])
                                        if len(real_corpus)>0:
                                            all_dig = True
                                            for pla in real_corpus:
                                                if re.search('[a-zA-Z]', pla):
                                                    all_dig = False
                                                    break
                                            if not all_dig:
                                                if (len(real_corpus) != 1) or is_arg or ((real_corpus[0] not in word_fre) and not hasNumbers(real_corpus[0]) and real_corpus[0] not in vocab_list):
                                                    real_corpus_dig_0 = []
                                                    for rpl in real_corpus:
                                                        if hasNumbers(rpl):
                                                            new_rpl = replace_digs(rpl)
                                                            groups = re.split('(\d+)',new_rpl)
                                                            real_corpus_dig_0.extend([t for t in groups if t])
#                                                            real_corpus_dig_0.append(new_rpl)
                                                        else:
                                                            real_corpus_dig_0.append(rpl)                                                                                                                       

                                                    #if real_corpus_dig_0 == ['duplin', 'county']:
                                                    # argument the highway place name
                                                    if bool_highway_check and is_highway:
                                                        illegar = True
                                                        if len(real_corpus_dig_0) >= 2:
                                                            for pp in real_corpus_dig_0:
                                                                if pp in highway_end or hasNumbers(pp):
                                                                    illegar = False
                                                                    break
                                                        if illegar:
                                                            random_index = random.sample(range(len(highway_value)), 1)
                                                            real_corpus_dig_0.append(highway_value[random_index[0]])
                                                    bool_remove = False
                                                    if bool_single_name_aug and is_arg:
                                                        if len(real_corpus_dig_0) == 1 and real_corpus_dig_0[0] in word_fre:

                                                            if row['type'] == 'city':
                                                                real_corpus_dig_0.append('city')
                                                            if row['type'] == 'town':
                                                                real_corpus_dig_0.append('town')
                                                            if row['type'] == 'suburb':
                                                                real_corpus_dig_0.append('suburb')
                                                            if row['type'] == 'county':
                                                                real_corpus_dig_0.append('county')

                                                        else:
                                                            for i in range(argu_count):
                                                                postive_add.append(tuple(real_corpus_dig_0))
    
                                                            if len(real_corpus_dig_0) >= 2 and real_corpus_dig_0[-1] in ['town','city', 'county','state']:
                                                                if len(real_corpus_dig_0) > 2 or real_corpus_dig_0[-2] not in word_fre:
                                                                    new_short = []
                                                                    for tt in range(len(real_corpus_dig_0)-1):
                                                                        new_short.append(real_corpus_dig_0[tt])
                                                                    for i in range(argu_count):
                                                                        postive_add.append(tuple(new_short))
                                                    else:
                                                         cc= 0
                                                    if not bool_remove:
                                                        '''agument the high level higway entity, such as I 39, an US 32 since they are important but less'''
                                                        if row['type'] == 'trunk' or row['type'] == 'motorway':
                                                            last_num_count += 1
                                                            for tt in range(argu_count):
                                                                high_way_augmented.append(tuple(real_corpus_dig_0))
                                                            '''agument the place entity with outofbag vocabulary'''
                                                        place_names.append(real_corpus_dig_0)
#                                                        if tuple(['cdt']) == tuple(real_corpus_dig_0):
#                                                            print(row)
                                                        if len(real_corpus_dig_0) == 1 and is_arg:
                                                            single_places.add(real_corpus_dig_0[0])
                                                        count += 1
                                        real_corpus= []
                                    else:
                                        if temps[i]:
                                            real_corpus.append(temps[i])
                            else:
                                real_corpus.append(x)
                        if len(real_corpus)>0:
                            all_dig = True
                            for pla in real_corpus:
                                if re.search('[a-zA-Z]', pla): #not pla.isdigit():
                                    all_dig = False
                                    break
                            if not all_dig:
                                if (len(real_corpus) != 1) or is_arg or ((real_corpus[0] not in word_fre) and not hasNumbers(real_corpus[0]) and real_corpus[0] not in vocab_list):
                                    real_corpus_dig_0 = []
                                    for rpl in real_corpus:
                                        if hasNumbers(rpl):
                                            new_rpl = replace_digs(rpl)
                                            groups = re.split('(\d+)',new_rpl)
                                            real_corpus_dig_0.extend([t for t in groups if t])
#                                            real_corpus_dig_0.append(new_rpl)
                                        else:
                                            real_corpus_dig_0.append(rpl)
                                     # argument the highway place name
                                    if bool_highway_check and is_highway:
                                        illegar = True
                                        if len(real_corpus_dig_0) >= 2:
                                            for pp in real_corpus_dig_0:
                                                if pp in highway_end or hasNumbers(pp):
                                                    illegar = False
                                                    break
                                        if illegar:
                                            random_index = random.sample(range(len(highway_value)), 1)
                                            real_corpus_dig_0.append(highway_value[random_index[0]])
                                    #if real_corpus_dig_0 == ['duplin', 'county']:
                                    #   pdb.set_trace()
                                    bool_remove = False
                                    if bool_single_name_aug and is_arg:
                                        #add type of place name for general words
                                        if len(real_corpus_dig_0) == 1 and real_corpus_dig_0[0] in word_fre:

                                            if row['type'] == 'city':
                                                real_corpus_dig_0.append('city')
                                            if row['type'] == 'town':
                                                real_corpus_dig_0.append('town')
                                            if row['type'] == 'suburb':
                                                real_corpus_dig_0.append('suburb')
                                            if row['type'] == 'county':
                                                real_corpus_dig_0.append('county')
                                        else:
                                            for i in range(argu_count):
                                                postive_add.append(tuple(real_corpus_dig_0))
                                            if len(real_corpus_dig_0) >= 2 and real_corpus_dig_0[-1] in ['town','city', 'county','state']:
                                                if len(real_corpus_dig_0) > 2 or real_corpus_dig_0[-2] not in word_fre:
                                                    new_short = []
                                                    for tt in range(len(real_corpus_dig_0)-1):
                                                        new_short.append(real_corpus_dig_0[tt])
                                                    for i in range(argu_count):
                                                        postive_add.append(tuple(new_short))
                                    else:
                                         cc =0
                                    if not bool_remove:
                                        '''agument the high level higway entity, such as I 39, an US 32 since they are important but less'''
                                        if row['type'] == 'trunk' or row['type'] == 'motorway':
                                            last_num_count += 1
                                            for tt in range(argu_count):
                                                high_way_augmented.append(tuple(real_corpus_dig_0))
                                        '''agument the place entity with outofbag vocabulary'''

                                        place_names.append(real_corpus_dig_0)
#                                        if tuple(['cdt']) == tuple(real_corpus_dig_0):
#                                            print(row)
                                        if len(real_corpus_dig_0) == 1 and is_arg:
                                            single_places.add(real_corpus_dig_0[0])

                                        count += 1
    print('count'+str(cc))
    print('last_num_count'+str(last_num_count))
    print('ofv_count'+str(ofv_count))
    add_names = load_osm_names(addition_file)
    if args.geonames:
        in_geonames_names = load_osm_names_fre('data/in_geonames.txt', word_fre, 1)
        us_geonames_names = load_osm_names_fre('data/us_geonames.txt', word_fre, 1)
        us_geonames_names.extend(in_geonames_names)
    else:
        us_geonames_names = load_osm_names_fre('data/extra_i_u.txt', word_fre, 1)

    word_fre_tuple = [tuple([word]) for word in word_fre]
    
    place_names_without_geonames = copy.deepcopy(place_names)
    place_names.extend(us_geonames_names)
    
    #get the distribution of length of words in gazetteers
#    word_l = {}
    for place in us_geonames_names:
        if len(place) in place_l.keys():
            place_l[len(place)] +=1
        else:
            place_l[len(place)] =1
    print(place_l)
#    return
#    place_names.extend(in_geonames_names)

    place_names.extend(add_names*argu_count*4)
    place_names.extend(state_augmented)
    #write_place('data/single_name.txt',single_place_names)
    write_place('data/osmnew1.txt',place_names)
    print(len(place_names))
    last_word_set = {};
    first_word_set = {};
    stop_word_thres = args.lt
    stop_word_thres_first = args.ft
    before_alb_word = set([])
    
    ''' count first and last words before augmentation'''
    for place in place_names:
        if not hasNumbers(place[-1]):
            if place[-1] in last_word_set.keys():
                last_word_set[place[-1]]+=1
            else:
                last_word_set[place[-1]]=1
        if not hasNumbers(place[0]):
            if place[0] in first_word_set.keys():
                first_word_set[place[0]]+=1
            else:
                first_word_set[place[0]]=1
        if place[-1].isalpha() and len(place[-1])==1:
            if len(place) > 1:
                before_alb_word.add(place[-2])
                
                
    '''replace the abbrevision words to generate new positive examples'''
    place_names.extend(high_way_augmented)
    place_names.extend(postive_add)

    aug_place_names, full, abbr = replace_abbrevison(place_names,file_name)

    set_place_names = set(aug_place_names)
    write_place('data/osmnames'+str(args.file)+'.txt',set_place_names)
    print('number of place names in OSM: ', len(set_place_names))
              
#    counter = collections.Counter(aug_place_names)
#    aug_place_names = list(set(aug_place_names).difference(set(word_fre_tuple)))
#    aug_place_names = copyACount(aug_place_names, counter)

    for i, fu in enumerate(full):
        ful = fu[-1]
        if ful in last_word_set.keys():
            if abbr[i] in last_word_set.keys():
                last_word_set[abbr[i]] += last_word_set[ful]
            else:
                last_word_set[abbr[i]] = last_word_set[ful]
                
    for i, fu in enumerate(full):
        ful = fu[-1]
        if ful in first_word_set.keys():
            if abbr[i] in first_word_set.keys():
                first_word_set[abbr[i]] += first_word_set[ful]
            else:
                first_word_set[abbr[i]] = first_word_set[ful]
                
    print(str(last_word_set['news'])+' news')
    print(str(last_word_set['nd'])+' nd')

    first_words = []
    last_words = []
    last_result_file = 'data/last.txt'
    last_result = open(last_result_file,'w')
    first_result_file = 'data/first.txt'
    first_result = open(first_result_file,'w')
    last_words_frequency = {}
    total_count_last = 0
    total_count_first = 0
    first_words_frequency = {}
    first_word_nonsense = []
    multiple_thres = 30
    multiple_first = args.multiple_f_l
    last_word_nonsense = []
    multiple_last = args.multiple_f_l


    for w in last_word_set.keys():
        last_result.write(w+' '+str(last_word_set[w]))
        last_result.write('\n')
        if last_word_set[w] >= stop_word_thres:
            last_words.append(w)
            last_words_frequency[w] = last_word_set[w]
            total_count_last += last_word_set[w]
#        else:
#            if last_word_set[w] > multiple_thres  and w not in single_places:
#                last_word_nonsense.extend([w]*last_word_set[w])
    for w in last_words_frequency.keys():
        last_words_frequency[w] = last_words_frequency[w]/float(total_count_last)

    for w in first_word_set.keys():
        first_result.write(w+' '+str(first_word_set[w]))
        first_result.write('\n')
        if first_word_set[w] >= stop_word_thres_first: #and first_word_set[w] in last_words:
            first_words.append(w)
            first_words_frequency[w] = first_word_set[w]
            total_count_first += first_word_set[w]
#        else:
#            if  first_word_set[w] > multiple_thres  and w not in single_places:
#                first_word_nonsense.extend([w]*first_word_set[w])
    for w in first_words_frequency.keys():
        first_words_frequency[w] = first_words_frequency[w]/float(total_count_first)


    if args.unseen:
        '''combine unseen words and category words to generate new positive examples'''
        unseen_positive = general_last_word2(last_words, unseen_words, 20)
        aug_place_names.extend(unseen_positive)
    result = sorted(set(map(tuple, aug_place_names)), reverse=True)
    ori_place_names = aug_place_names
    
    '''get the prefix words of numbers in positive examples'''
    after_num_words = []
    before_num_words = []
    for place in aug_place_names:
        for i, plw in enumerate(place):
            if hasNumbers(plw):
                if i != 0:
                    before_num_words.append(tuple([place[i-1]]))
                if i != len(place)-1:
                    after_num_words.append(tuple([place[i+1]]))
                front, back = split_numbers(plw)
                if front:
                    before_num_words.append(tuple([front]))
                if back:
                    after_num_words.append(tuple([back]))
    positive_number_prefixs = after_num_words
    positive_number_prefixs.extend(before_num_words)
    after_num_words_set = set(after_num_words)

    '''get the prefix words of numbers in negative examples'''
    aug_place_names = result
#    neg_num_prefix = list(set(glove_num_prefix).difference(set(positive_number_prefixs)))
    neg_num_prefix = list(set(glove_num_prefix))
    ''' add the number prefix word from the osm to that of negative ones 
    from google embedding, such as the highway''' 
    inserted_neg = []
    
    '''combine the prefix word (e.g., ft) with numbers (e.g., 000) to form new negative examples such as [ft 000] and [00 ft]'''
    negative_prefix_places = negative_prefix_num(neg_num_prefix, 40)
    insert_negative_prefix_places = copy.deepcopy(negative_prefix_places)
    counter = collections.Counter(insert_negative_prefix_places)
    insert_negative_prefix_places = list(set(insert_negative_prefix_places).difference(set(aug_place_names)))
    insert_negative_prefix_places = copyACount(insert_negative_prefix_places, counter)
    inserted_neg.extend(insert_negative_prefix_places)
    print('insert_negative_prefix_places count: ' + str(len(insert_negative_prefix_places)))

    ''' extract the sub version of the positive examples as negative examples 
    if the end of the example is not place category name, such as street'''
    osm_gen_voc_fre = 10
    real_first_words = set(first_words).intersection(set(last_words))
    real_first_words = list(set(real_first_words).difference(set(list(string.ascii_lowercase))))
    if args.bool_rm_pla_f:
        real_first_words = list(set(real_first_words).difference(set(single_places)))
    print(real_first_words)
    
    short_negs, single_neg = short_neg(aug_place_names, last_words, real_first_words)
    if args.bool_short:
        counter=collections.Counter(short_negs)
        short_negs = list(set(short_negs).difference(set(aug_place_names)))
        short_negs = copyACount(short_negs, counter)
        inserted_neg.extend(short_negs)
        print('short_negs count: ' + str(len(short_negs)))
    
    '''create the general list from the gazetteer, ascii, number-prefix words, numbers, and pre-embedding vocubularies'''
    #get the general words from gazetteer 
    counter=collections.Counter(single_neg)
    single_neg = list(set(single_neg).difference(set(aug_place_names)))
    single_neg = copyACount(single_neg, counter,osm_gen_voc_fre)
    single_neg_g_v_l = 200000
    if single_neg_g_v_l > len(single_neg):
        single_neg_g_v_l = len(single_neg)
    random_indexs = random.sample(range(len(single_neg)), single_neg_g_v_l)
    single_neg =  [single_neg[i][0] for i in random_indexs]
    #add the ascii chars into general word list  
    ascii_chars = list(string.ascii_lowercase)*100
    ascii_chars_set =  [(x,) for x in list(string.ascii_lowercase)]
    vocab_list.extend(ascii_chars)
    #add the google embedding vocabularies into general word list  
    vocab_list.extend(google_vocabs)     
    #add the single words from positive examples to the general word list
    vocab_list.extend(single_neg)
    vocab_list.extend(last_word_nonsense)
    vocab_list.extend(first_word_nonsense)

    #add the number-prefix words to the general word list
    neg_num_prefix = [w[0] for w in neg_num_prefix]
    vocab_list.extend(neg_num_prefix*60)
    #add the numbers words to the general word list

    num_gen_vocabs = []
    num_gen_vocabs_count = 200
    num_gen_vocabs.extend(['0']*num_gen_vocabs_count)
    num_gen_vocabs.extend(['00']*num_gen_vocabs_count)
    num_gen_vocabs.extend(['000']*num_gen_vocabs_count)
    num_gen_vocabs.extend(['0000']*num_gen_vocabs_count)
    num_gen_vocabs.extend(['00000']*num_gen_vocabs_count)
    num_gen_vocabs.extend(['000000']*num_gen_vocabs_count)
    num_gen_vocabs.extend(['0000000']*num_gen_vocabs_count)
    num_gen_vocabs.extend(['00000000']*num_gen_vocabs_count)
    num_gen_vocabs.extend(['000000000']*num_gen_vocabs_count)
    num_gen_vocabs.extend(['0000000000']*num_gen_vocabs_count)
    num_gen_vocabs.extend(['00000000000']*num_gen_vocabs_count)

    vocab_list.extend(num_gen_vocabs)

    num_gen_vocabs2 = []
    num_gen_wocads_count2 = 50
    num_gen_vocabs2.extend(['0']*num_gen_wocads_count2)
    num_gen_vocabs2.extend(['00']*num_gen_wocads_count2)
    num_gen_vocabs2.extend(['000']*num_gen_wocads_count2)
    num_gen_vocabs2.extend(['0000']*num_gen_wocads_count2)
    num_gen_vocabs2.extend(['00000']*num_gen_wocads_count2)

    '''randomly combine general words to form the negative samples'''
    gen_count = 50000000
    neg_gens = general_words_neg(vocab_list, gen_count, last_words, real_first_words, general_words_first)
    neg_gens = list(set(neg_gens).difference(set(aug_place_names)))
    counter=collections.Counter(neg_gens)
    neg_gens = copyACount(neg_gens, counter)
    inserted_neg.extend(neg_gens)
    print('neg_gens count: ' + str(len(neg_gens)))

    '''general words are treated as negative samples'''
    new_vocab_tuple = [(x,) for x in vocab_list]
    counter = collections.Counter(new_vocab_tuple)
    general_neg = list(set(new_vocab_tuple).difference(set(aug_place_names)))
    general_neg = copyACount(general_neg, counter)
    inserted_neg.extend(general_neg)
    print('general_neg count: ' + str(len(general_neg)))

    '''insert the most general words before the place category words to generate negative examples, 
    for instance, 'on the way' and 'at the street' will be generated'''

    general_last_words_neg = general_last_word_adv(last_words, last_words_frequency, most_general_words, 50000000)
    write_place('data/gene_before.txt',general_last_words_neg)
    counter = collections.Counter(general_last_words_neg)
    general_last_words_neg = list(set(general_last_words_neg).difference(set(aug_place_names)))
    general_last_words_neg = copyACount(general_last_words_neg, counter)
    write_place('data/gene_after.txt',general_last_words_neg)
    inserted_neg.extend(general_last_words_neg)
    print('general_last_words_neg count: ' + str(len(general_last_words_neg)))

    '''remove the imvalid place names in aug_place_names such as the street, 
    0 street altuhouth they are on the gazetteer'''
    new_most_general_words = most_general_words
    new_most_general_words.append('0')
    new_most_general_words.append('00')
    new_most_general_words.append('000')
    new_most_general_words.append('0000')
    new_last_words2 = last_words
#    new_last_words2.append('the')
    new_last_words2.append('th')
    new_last_words2.append('nd')
    general_last_words_neg2 = general_last_word2(new_last_words2, new_most_general_words, 50)
    counter = collections.Counter(ori_place_names)
    ori_place_names = list(set(ori_place_names).difference(set(general_last_words_neg2)))
    ori_place_names = copyACount(ori_place_names, counter)
    inserted_neg.extend(general_last_words_neg2)
    print('general_last_words_neg2 count: ' + str(len(general_last_words_neg2)))

    
    new_last_words = last_words
    new_last_words.append('0')
    new_last_words.append('00')
    new_last_words.append('000')
    new_last_words.append('0000')
    general_last_words_neg3 = general_last_word2(new_last_words, more_general_words, 5)
    counter = collections.Counter(general_last_words_neg3)
    general_last_words_neg3 = list(set(general_last_words_neg3).difference(set(aug_place_names)))
    general_last_words_neg3 = copyACount(general_last_words_neg3, counter)
    inserted_neg.extend(general_last_words_neg3)
    print('general_last_words_neg3 count: ' + str(len(general_last_words_neg3)))

    if args.bool_insert_first:
        '''insert the most general words before a first category word'''
#        new_first_words = list(set(first_words).difference(set(single_places)))
        cur_more_general_words = list(set(more_general_words).difference(set(last_words)))
        general_first_words_neg3 = general_first_word2(real_first_words, cur_more_general_words, 5)
        counter = collections.Counter(general_first_words_neg3)
        general_first_words_neg3 = list(set(general_first_words_neg3).difference(set(aug_place_names)))
        general_first_words_neg3 = copyACount(general_first_words_neg3, counter)
        inserted_neg.extend(general_first_words_neg3)
        print('general_first_words_neg3 count: ' + str(len(general_first_words_neg3)))
        
        cur_most_general_words= list(set(most_general_words).difference(set(last_words)))
        general_first_words_neg = general_first_word_adv(real_first_words, first_words_frequency, cur_most_general_words, 10000000)
        counter = collections.Counter(general_first_words_neg)
        general_first_words_neg = list(set(general_first_words_neg).difference(set(aug_place_names)))
        general_first_words_neg = copyACount(general_first_words_neg, counter)
        inserted_neg.extend(general_first_words_neg)
        print('general_first_words_neg count: ' + str(len(general_first_words_neg)))


    if args.bool_disorder:
        '''change the order of the tokens in the positive examples'''
        disorder_negs = adjust_order_neg(aug_place_names, last_words)
        counter=collections.Counter(disorder_negs)
        disorder_negs = list(set(disorder_negs).difference(set(aug_place_names)))
        disorder_negs = copyACount(disorder_negs, counter)
        inserted_neg.extend(disorder_negs)
        print('disorder_negs count: ' + str(len(disorder_negs)))


    counter = collections.Counter(rem_sym_negs)
    rem_sym_negs = list(set(rem_sym_negs).difference(set(aug_place_names)))
    rem_sym_negs = copyACount(rem_sym_negs, counter)
    inserted_neg.extend(rem_sym_negs)
    print('rem_sym_negs count: ' + str(len(rem_sym_negs)))

    '''randomly choose a coupl of general words and insert them in 
       the first and last of a positive example'''
    random_place_names = random.sample(aug_place_names, int(len(aug_place_names)/3))       
    set_before_num_words = set(before_num_words)
    temp_neg = get_neg_insert(random_place_names, vocab_list, set_before_num_words, last_words, real_first_words, count=4)
    inserted_neg.extend(temp_neg)
    print('temp_neg count: ' + str(len(temp_neg)))

    
    '''combine ascii chars and numbers'''
    ascii_num_negs = ascii_num_neg(50)
    counter = collections.Counter(ascii_num_negs)
    ascii_num_negs = list(set(ascii_num_negs).difference(set(aug_place_names)))
    ascii_num_negs = copyACount(ascii_num_negs, counter)
    inserted_neg.extend(ascii_num_negs)
    print('ascii_num_negs count: ' + str(len(ascii_num_negs)))

    '''insert general words to the entity of negative_prefix_places, which consists of numbers and prefix words'''
    inserted_neg_prefix =  get_neg_insert_prex_gen(negative_prefix_places, vocab_list) 
    counter = collections.Counter(inserted_neg_prefix)
    inserted_neg_prefix = list(set(inserted_neg_prefix).difference(set(aug_place_names)))
    inserted_neg_prefix = copyACount(inserted_neg_prefix, counter)
    inserted_neg.extend(inserted_neg_prefix)
    print('inserted_neg_prefix count: ' + str(len(inserted_neg_prefix)))

    '''insert most gnereal words and place category words to the entity of negative_prefix_places, which consists of numbers and prefix words'''
    gen_last_mix_words = []
    gen_last_mix_words.extend(last_words)
    gen_last_mix_words.extend(most_general_words)
    inserted_neg_prefix_2 =  get_neg_insert_prex_gen(negative_prefix_places, gen_last_mix_words) 
    counter = collections.Counter(inserted_neg_prefix_2)
    inserted_neg_prefix_2 = list(set(inserted_neg_prefix_2).difference(set(aug_place_names)))
    inserted_neg_prefix_2 = copyACount(inserted_neg_prefix_2, counter)
    inserted_neg.extend(inserted_neg_prefix_2)
    print('inserted_neg_prefix_2 count: ' + str(len(inserted_neg_prefix_2)))

    '''randomly combine numbers'''
    neg_gens_dig = general_words_neg_dig(num_gen_vocabs, 3000000)
    num_gen_vocabs2 = []
    num_gen_wocads_count2 = 10
    num_gen_vocabs2.extend(['0']*num_gen_wocads_count2)
    num_gen_vocabs2.extend(['00']*num_gen_wocads_count2)
    num_gen_vocabs2.extend(['000']*num_gen_wocads_count2)
    num_gen_vocabs2.extend(['0000']*num_gen_wocads_count2)

    '''randomly combine numbers and general words'''

    neg_dig_names = get_neg_insert_random(neg_gens_dig, vocab_list, insert_word_len = 4)
    counter = collections.Counter(neg_dig_names)
    neg_dig_names = list(set(neg_dig_names).difference(set(aug_place_names)))
    neg_dig_names = copyACount(neg_dig_names, counter)
    inserted_neg.extend(neg_dig_names)
    print('neg_dig_names count: ' + str(len(neg_dig_names)))
    
    '''randomly combine numbers and ascill chars'''
    ascii_chars.extend(num_gen_vocabs2)
    asica_neg_gens = general_words_neg_dig(ascii_chars, 4000000)
    counter = collections.Counter(asica_neg_gens)
    asica_neg_gens = list(set(asica_neg_gens).difference(set(aug_place_names)))
    asica_neg_gens = copyACount(asica_neg_gens, counter)
    inserted_neg.extend(asica_neg_gens)
    print('asica_neg_gens count: ' + str(len(asica_neg_gens)))

    '''randomly choose a coupl of general words and insert them in 
       the last of a positive example'''
    random_place_names = random.sample(aug_place_names, int(len(aug_place_names)/3))       
    neg_insert_last = get_neg_insert_last(random_place_names, vocab_list, last_words, count=10)
    counter = collections.Counter(neg_insert_last)
    neg_insert_last = list(set(neg_insert_last).difference(set(aug_place_names)))
    neg_insert_last = copyACount(neg_insert_last, counter)
    inserted_neg.extend(neg_insert_last)
    print('neg_insert_last count: ' + str(len(neg_insert_last)))

    '''randomly choose a coupl of general words and insert them in 
       the first of a positive example'''
#    if args.south_ignore:
#        ignore_first_words = ['west','north','east','south','northeast','southeast','northwest','southwest','central'];
#    else:
    ignore_first_words = []; 
    ignore_first_words.extend(unseen_words)
    random_place_names = random.sample(aug_place_names, int(len(aug_place_names)/3))
    neg_insert_first = get_neg_insert_first(random_place_names, vocab_list, ignore_first_words, count=2)
    counter = collections.Counter(neg_insert_first)
    neg_insert_first = list(set(neg_insert_first).difference(set(aug_place_names)))
    neg_insert_first = copyACount(neg_insert_first, counter)
    inserted_neg.extend(neg_insert_first)
    print('neg_insert_first count: ' + str(len(neg_insert_first)))
   
    '''insert numbers in the end of a positive name'''
    neg_insert_last_num = get_neg_insert_alpha(aug_place_names, num_gen_vocabs, after_num_words_set, 10000000)
    counter = collections.Counter(neg_insert_last_num)
    neg_insert_last_num = list(set(neg_insert_last_num).difference(set(aug_place_names)))
    neg_insert_last_num = copyACount(neg_insert_last_num, counter)
    inserted_neg.extend(neg_insert_last_num)
    print('neg_insert_last_num count: ' + str(len(neg_insert_last_num)))

    '''insert numbers in the begining of a positive name'''
    neg_insert_first_num = get_neg_insert_first_num(aug_place_names, num_gen_vocabs, set_before_num_words, 10000000)
    counter = collections.Counter(neg_insert_first_num)
    neg_insert_first_num = list(set(neg_insert_first_num).difference(set(aug_place_names)))
    neg_insert_first_num = copyACount(neg_insert_first_num, counter)
    inserted_neg.extend(neg_insert_first_num)
    print('neg_insert_first_num count: ' + str(len(neg_insert_first_num)))

    '''insert prefix words and numbers in the begining and end of a positive name'''
    neg_insert_prex_num = get_neg_insert_prex_num(aug_place_names, negative_prefix_places, 5000000)
    counter = collections.Counter(neg_insert_prex_num)
    neg_insert_prex_num = list(set(neg_insert_prex_num).difference(set(aug_place_names)))
    neg_insert_prex_num = copyACount(neg_insert_prex_num, counter)
    inserted_neg.extend(neg_insert_prex_num)
    print('neg_insert_prex_num count: ' + str(len(neg_insert_prex_num)))
    
    '''insert ascii char in the begining of a positive name'''
    neg_insert_alpha = get_neg_insert_alpha(aug_place_names, list(string.ascii_lowercase), list(before_alb_word), max_count=6000000)
    counter = collections.Counter(neg_insert_alpha )
    neg_insert_alpha = list(set(neg_insert_alpha ).difference(set(aug_place_names)))
    neg_insert_alpha = copyACount(neg_insert_alpha, counter)
    inserted_neg.extend(neg_insert_alpha)
    print('neg_insert_alpha count: ' + str(len(neg_insert_alpha)))

    if bool_alpha_gen:
        neg_insert_alpha_gen = get_neg_insert_alpha_gen(new_vocab_tuple, list(string.ascii_lowercase), list(before_alb_word), max_count=6000000)
        counter = collections.Counter(neg_insert_alpha_gen )
        neg_insert_alpha = list(set(neg_insert_alpha_gen).difference(set(aug_place_names)))
        neg_insert_alpha = copyACount(neg_insert_alpha_gen, counter)
        inserted_neg.extend(neg_insert_alpha_gen)
        print('neg_insert_alpha_gen count: ' + str(len(neg_insert_alpha_gen)))

    '''insert general words between two positive examples'''
    if bool_comb_neg_gen:
        combine_neg_gen_places = combine_neg_gen(aug_place_names, most_gen_words_2, max_count=10000000*args.multiple_f_l)
        counter = collections.Counter(combine_neg_gen_places)
        combine_neg_gen_places = list(set(combine_neg_gen_places).difference(set(aug_place_names)))
        combine_neg_gen_places = copyACount(combine_neg_gen_places, counter)
        inserted_neg.extend(combine_neg_gen_places)
        print('combine_neg_gen_places count: ' + str(len(combine_neg_gen_places)))
 
    if not args.unseen:
        '''combine unseen words and category words to generate new positive examples'''
        unseen_positive = general_last_word2(last_words, unseen_words, 20)
        ori_place_names.extend(unseen_positive)
        unseen_negative = []
        for i in range(1000):
            unseen_negative.append(tuple(unseen_words))
        inserted_neg.extend(unseen_negative)
        print('unseen_negative count: ' + str(len(unseen_negative)))

    '''save positive and negative examples'''
        
    print('positive count: ' + str(len(ori_place_names)))
    lr_p = random.sample(ori_place_names, len(ori_place_names))
    write_place(pos_file, lr_p)

    print('negative count: ' + str(len(inserted_neg)))
    print(len(inserted_neg))
    
    multi = 5
    if int(multi*len(ori_place_names)) > len(inserted_neg):
        sample_length = len(inserted_neg)
    else:
        sample_length = int(multi*len(ori_place_names))
    lr = random.sample(inserted_neg, sample_length)
    write_place(neg_file,lr)

    lr = random.sample(inserted_neg, len(inserted_neg))
    write_place(full_neg_file,lr)
    print('total time:', time.time()-start_time)
if __name__ == '__main__':
    main()
