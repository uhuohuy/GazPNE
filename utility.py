import re
import codecs

'''replace the number of a string by 0, such as hwy12 to hwy00'''
def replace_digs(word):
    new_word = ''
    for i, c in enumerate(word):
        if c.isdigit():
            new_word+='0'
        else:
            new_word+=c
    return new_word
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

'''load place names from a file'''    
def load_osm_names_fre(pos_f, fre_words, aug_count = 1):
    pos_training_data = []
    with codecs.open(pos_f, 'r',encoding='utf-8') as file:
        for line in file:
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
    return pos_training_data

'''load place names from a file'''    

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

if __name__ == '__main__':
    test = '1233fc55tg- f'
    print(split_numbers(test))
    print(re.findall(r'[A-Za-z]|-?\d+\.\d+|\d+',test))
    res = [re.findall(r'(\w+?)(\d+)', test)[0] ]
    groups = re.split('(\d+)',test)
    print(replace_digs('5578sfhfhjf22'))
