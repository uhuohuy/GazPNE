import re
import codecs
from constant import POS_TAGS
import unicodedata
import csv
import numpy as np
import re
import pdb

def index_list_int(index, index_list, sub_index):
    bool_intersect = 0
    for i in index_list:
        if intersection(sub_index[index],sub_index[i]):
            bool_intersect = 1
            break
    return bool_intersect

def index_list_sub(index, index_list, sub_index):
    return_sub = []
    for i in index_list:
        if is_Sublist(sub_index[index],sub_index[i]):
            return_sub.append(i)
        elif is_Sublist(sub_index[i],sub_index[index]):
            return_sub.append(index)
    return list(set(return_sub))

def offinoffset(cur_off_pla, hashtag_offsets):
    for off in hashtag_offsets:
        if cur_off_pla[0] >= off[0] and cur_off_pla[0] <= off[1] and \
           cur_off_pla[1] >= off[0] and cur_off_pla[1] <= off[1]:
               return True
    return False


def interset_adv(list1,list2):
    first_place = ''
    second_place = ''
    for i in list1:
        first_place += i.lower()
    for j in list2:
        second_place += j.lower()
    if first_place==second_place:
        match = 1
    else:
        match = 0
    return match

''' judge if two list has shared elements  '''
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3 

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

def extract_place(pd_item):
    return_places = []
    for item in pd_item:
        if '-' not in item and '/' not in item and '(' not in item:
            item = unicodedata.normalize('NFKD', item).encode('ascii','ignore').decode("utf-8") 
            row_nobrackets = re.sub("[\(\[].:;*?[\)\]]", "", item)
            corpus = [word.lower() for word in re.split("[. #,&\"\',’]",row_nobrackets)]
            new_corpus = []
            for cor in corpus:
            	all_ascii = ''.join(char for char in cor if ord(char) < 127)
            	new_corpus.append(all_ascii)
            corpus = [x for x in new_corpus if x and (len(x) < 2 or (len(x)>=2 and not (x[0]== '(' and x[len(x)-1]== ')')))]
            if corpus:
                return_places.append(tuple(corpus))
    return return_places

def extract_tokens(pos_f):
    very_fre_words = []
    with codecs.open(pos_f, 'r',encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = re.split("\s",line)
            tokens = list(filter(None, tokens))
            tokens = [x.lower() for x in tokens if x]
            very_fre_words.extend(tokens)
    return list(set(very_fre_words))


'''replace the number of a string by 0, such as hwy12 to hwy00'''
def replace_digs(word):
    new_word = ''
    for i, c in enumerate(word):
        if c.isdigit():
            new_word+='0'
        else:
            new_word+=c
    return new_word

def abbrevison1(abr_file):
    with open(abr_file,mode='r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        abbr = {}
        for row in reader:
            tokens = row[0].split(' ')
#            if len(tokens) == 1:
            abbr[row[1]] = tokens 
    return abbr

def abbrevison(abr_file):
    with open(abr_file,mode='r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        abbr = {}
        for row in reader:
            tokens = row[0].split(' ')
            if len(tokens) == 1:
                abbr[row[1]] = tokens[0] 
    return abbr


def pt2vector(tags):
    vec = []
    for tag in tags:
        if tag[1] not in POS_TAGS.keys():
            break
        tag_id = POS_TAGS[tag[1]]
        zero_list = [0]*len(POS_TAGS)
        zero_list[tag_id-1] = 1
        vec.extend(zero_list)
    return vec

'''to judege if a string contains numbers'''
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

'''write list of place names into file'''
def write_place(file_name, place_names):
    f= codecs.open(file_name,"w+")
    for neg in place_names:
        temp= ''
        for negraw in neg:
            temp = temp+negraw+' ' 
        f.write(temp+'\n')
    f.close()

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

'''load place names from a file'''    
def load_osm_names_fre(pos_f, fre_words, aug_count = 1, return_gen = 0):
    pos_training_data = []
    general_places = set()
    with codecs.open(pos_f, 'r',encoding='utf-8') as file:
        for line in file:
            line = unicodedata.normalize('NFKD', line).encode('ascii','ignore').decode("utf-8") 
            line = line.strip()
            if len(line) == 0:
                continue
            row_nobrackets = re.sub("[\(\[].:;*?[\)\]]", "", line)         
            corpus = [word.lower() for word in re.split("[. #,&\"\',’]",row_nobrackets)]
            corpus = [word  for word in corpus if word]
            corpus = [replace_digs(word) for word in corpus]
            final_result = []
            for token in corpus:
                groups = re.split('(\d+)',token)
                groups = [g for g in groups if g]
                final_result.extend(groups)
            if not(len(final_result) == 1 and final_result[0] in fre_words):
                for k in range(aug_count):
                    pos_training_data.append(tuple(final_result))
            else:
                if return_gen:
                    general_places.add(tuple(final_result))
    if return_gen:
        return pos_training_data, general_places
    else:
        return pos_training_data

'''load place names from a file'''    
def string2pla(rawstr):
    line = rawstr.strip()
    row_nobrackets = re.sub("[\(\[].:;*?[\)\]]", "", line)         
    corpus = [word.lower() for word in re.split("[. #,&\"\',’]",row_nobrackets)]
    corpus = [word  for word in corpus if word]
#    tokens = line.split(' ')
    return tuple(corpus)

def load_osm_names(pos_f):
    pos_training_data = []
    with codecs.open(pos_f, 'r',encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split(' ')
            pos_training_data.append(tuple(tokens[0:len(tokens)])) 
    return pos_training_data

isascii = lambda s: len(s) == len(s.encode())

'''split a word to multiple sub words by numbers'''    
def split_numbers(word):
    groups = re.split('(\d+)',word)
    num_tag = False
    front_word = ''
    back_word = ''
    for g in groups:
        if hasNumbers(g):
            num_tag = True
        else:
            new_word = "".join(re.findall("[a-zA-Z]*", g))
            if new_word:
                if num_tag :
                    back_word = new_word
                    break
                else:
                    front_word = new_word

    return front_word, back_word
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

if __name__ == '__main__':
#    read_ent_file('data/word_ent100_copy.txt')
    print(extract_tokens('data/fc.txt'))
    print(index_list_int(1,[0,2,3],[[0,1,2],[0,1,3],[0,1,2,3,4,],[0,1]]))
    print(index_list_sub(1,[0,2,3],[[0,2],[0,1,3],[0,1,3,4],[0,1]]))

    #    print(string2pla('us flood.'))
#    test = '1233fc55tg- f'
#    print(split_numbers(test))
#    print(re.findall(r'[A-Za-z]|-?\d+\.\d+|\d+',test))
#    res = [re.findall(r'(\w+?)(\d+)', test)[0] ]
#    groups = re.split('(\d+)',test)
#    print(replace_digs('5578sfhfhjf22'))
#    print(softmax([0.91,0.03,0.05,0.01]))
#    item= "RT @iH8TvvitterHoes: Nigga that's Nuketown Rtì@HistoryInPix: The Great Alaska Earthquake of 1964 http://t.co/CGQzLUahHUî"
#    item = unicodedata.normalize('NFKD', item).encode('ascii','ignore').decode("utf-8") 
#    print(item)
#if __name__ == '__main__':
#    # main()
#    print(softmax([0.91,0.03,0.05,0.01]))
