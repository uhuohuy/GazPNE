'''#############################################################################
Copyright 2017 Hussein S. Al-Olimat, hussein@knoesis.org

This software is released under the GNU Affero General Public License (AGPL)
v3.0 License.
#############################################################################'''

import re
import string
import itertools
import collections
import unicodedata
from itertools import groupby
from wordsegment import load, segment
from operator import itemgetter
from collections import defaultdict
from spellchecker import SpellChecker
from utility import hasNumbers, replace_digs,abbrevison1
import copy
import pdb
load()
spell = SpellChecker()

import Twokenize

################################################################################
################################################################################

__all__ = [ 'set_global_env',
            'Stack',
            'Tree',
            'preprocess_tweet',
            'flatten',
            'build_tree',
            'using_split2',
            'findall',
            'align_and_split',
            'extract',
            'do_they_overlap',
            'filterout_overlaps',
            'find_ngrams',
            'remove_non_full_mentions',
            'init_Env',
            'initialize']

################################################################################

printable = set(string.printable)
exclude = set(string.punctuation)

url_re = r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*' # urls regular expression
mentions_re = r"@[\w\.]+" # mentions regular expression

# LNEx global environment
env = None

def set_global_env(g_env):
    '''Sets the global environment at the time of initialization to be used
    module-wide'''

    global env
    env = g_env

cap_word_shape = False

################################################################################

class Stack(object):
    '''Stack data structure'''

    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop(0)

    def notEmpty(self):
        return self.items != []

################################################################################

class Tree(object):
    '''Tree data structure used for building the bottom up tree of n-grams'''

    def __init__(self, cargo, tokensIndexes, left=None, right=None, level=1):

        # the data of the node. i.e., the n-gram tokens
        self.cargo = cargo
        # n-gram tokens indexes from the original tweet text
        self.tokensIndexes = tokensIndexes
        # left child of tree node
        self.left = left
        # right child of tree node
        self.right = right
        # unigrams are level 1 -> bigrams are level 2
        self.level = level

    def __str__(self):
        return str(self.cargo)

################################################################################

def strip_non_ascii(s):
    nfkd = unicodedata.normalize('NFKD', s)
    return str(nfkd.encode('ASCII', 'ignore').decode('ASCII'))

def get_removed_indices(tweet):
    # Contains the indices of characters that were removed from the oringial text
    removedIndices = set()

    for r in [url_re, mentions_re]:
        for m in [(m.start(),m.end()) for m in re.finditer(r, tweet)]:
            # add all character offsets to the set of removed indices
            if r in [mentions_re]:
                removedIndices.update(set(range(m[0]+1,m[1])))
            else:
                removedIndices.update(set(range(m[0],m[1])))

    return removedIndices

def preprocess_tweet(tweet,word_list, remove_men=1):
    '''Preprocesses the tweet text and break the hashtags'''

    # remove retweet handler
    if tweet[:2] == "rt":
        try:
            colon_idx = tweet.index(": ")
            tweet = tweet[colon_idx + 2:]
        except BaseException:
            pass

    # remove url from tweet
    tweet = re.sub(url_re, '', tweet)

    # remove non-ascii characters
    tweet = "".join([x for x in tweet if x in printable])

    # additional preprocessing
    tweet = tweet.replace("\n", " ").replace(" https", "").replace("http", "")

    # remove all mentions
    if remove_men:
        tweet = re.sub(mentions_re, " @ ", tweet)

    # extract hashtags to break them -------------------------------------------
    hashtags = re.findall(r"#\w+", tweet)

    # This will contain all hashtags mapped to their broken segments
    # e.g., #ChennaiFlood : {chennai, flood}
    replacements = defaultdict()

    for hashtag in hashtags:

        # keep the hashtag with the # symbol
        _h = hashtag[1:]

        # remove any punctuations from the hashtag and mention
        # ex: Troll_Cinema => TrollCinema
        _h = _h.translate(str.maketrans('','',''.join(string.punctuation)))

        # breaks the hashtag
        if _h.lower() not in word_list:
            segments = segment(_h)
            start_p = 0
            for i, seg in enumerate(segments):
                segments[i]=_h[start_p:start_p+len(seg)]
                start_p=start_p+len(seg)
        else:
            segments = [_h]
        # concatenate the tokens with spaces between them
        segments = ' '.join(segments)
        segments = ' # ' + segments
        replacements[hashtag] = segments

    # replacement of hashtags in tweets
    # e.g., if #la & #laflood in the same tweet >> replace the longest first
    for k in sorted(
            replacements,
            key=lambda k: len(
                replacements[k]),
            reverse=True):
        tweet = tweet.replace(k, replacements[k])

    # --------------------------------------------------------------------------

    # padding punctuations
#    tweet = re.sub('([,!$%|?():])', r' \1 ', tweet)
    tweet = re.sub('([,$|%!?/&():])', r' \1 ', tweet)
#    pdb.set_trace()
#    tweet = tweet.replace("-", " ")#
    tweet = tweet.replace(" 's", " ")#    
#    tweet = tweet.replace(". ", " . ").replace("-", " ")#
    # shrink blank spaces in preprocessed tweet text to only one space
    tweet = re.sub('\s{2,}', ' ', tweet)

    # # remove consecutive duplicate tokens which causes an explosion in tree
    # while re.search(r'\b(.+)(\s+\1\b)+', tweet):
    #     tweet = re.sub(r'\b(.+)(\s+\1\b)+', r'\1', tweet)

    # remove trailing spaces
    tweet = tweet.strip()

    return tweet

################################################################################
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

def flatten(l):
    '''Flattens a list of lists to a list.
    Based on Cristian answer @ http://stackoverflow.com/questions/2158395'''

    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

################################################################################

def build_tree(glm, ts):
    ''' Build a bottom-up tree of valid ngrams using the gazetteer langauge
    model (glm) and the tweet segment (ts)'''

    # dictionary of valid n-grams
    valid_n_grams = defaultdict(float)

    # store tree nodes in a stack
    nodes = Stack()

    # init leaf nodes from the unigrams
    for index, token in enumerate(ts):
        nodes.push(
            Tree( cargo=token,
                  tokensIndexes=set([index]),
                  left=None,
                  right=None,
                  level=1))

    node1 = None
    node2 = None

    while nodes.notEmpty():

        if node1 is None:
            node1 = nodes.pop()
            node2 = nodes.pop()

        else:
            node1 = node2
            node2 = nodes.pop()

        # process nodes of similar n-gram degree/level
        if node1.level == node2.level:

            _node2 = node2

            # if there is a common child take only the right child
            if node1.right == node2.left and node1.right is not None:
                _node2 = node2.right

            tokens_list = list()

            # find valid n-grams from the cartisian product of two tree nodes
            for i in itertools.product(node1.cargo, _node2.cargo):

                # flatten the list of lists
                flattened = list(flatten(i))

                # remove consecutive duplicates
                final_list = list(map(itemgetter(0), groupby(flattened)))

                # prune based on the probability from the language model
                p = " ".join(final_list)
                p = p.strip()

                # the probability of a phrase p calculated by the language model
                score = glm.phrase_probability(p)

                # if an n-gram is valid then add it to the dictionary
                if score > 0:

                    # e.g., ('avadi', 'rd', (4, 5))
                    valid_n_gram = tuple(final_list) + \
                                    (tuple(set(node1.tokensIndexes | \
                                    _node2.tokensIndexes)),)

                    # e.g., ('avadi', 'rd', (4, 5)) 5.67800416892e-05
                    valid_n_grams[valid_n_gram] = score

                    # the cargo of the node
                    tokens_list.append(final_list)

            # create higher level nodes and store them in the stack
            nodes.push(
                Tree(
                    tokens_list,
                    (node1.tokensIndexes | node2.tokensIndexes),
                    node1,
                    node2,
                    (node1.level + node2.level)))

    return valid_n_grams

################################################################################

def using_split2(line, _len=len):
    '''Tokenizes the tweet and retain the offsets of each token
    Based on aquavitae answer @ http://stackoverflow.com/questions/9518806/'''

    words = Twokenize.tokenize(line)
    index = line.index
    offsets = []
    append = offsets.append
    running_offset = 0
    for word in words:
        word_offset = index(word, running_offset)
        word_len = _len(word)
        running_offset = word_offset + word_len
        append((word, word_offset, running_offset - 1))
    return offsets

################################################################################

#
def findall(p, s):
    '''Yields all the positions of the pattern p in the string s.
    Based on AkiRoss answer @ http://stackoverflow.com/questions/4664850'''

    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i + 1)

################################################################################

def align_and_split(raw_string, preprocessed_string):
    '''Aligns the offsets of the preprocessed tweet with the raw tweet to retain
    original offsets when outputing spotted location names'''

    removedIndices = get_removed_indices(raw_string)

    tokens = list()

    last_index = 0

    for token in using_split2(preprocessed_string):

        matches = [(raw_string[i:len(token[0]) + i], i, len(token[0]) + i - 1)
                   for i in findall(token[0], raw_string)]

        for match in matches:
            if match[1] >= last_index and match[1] not in removedIndices:
                last_index = match[2]
                tokens.append(match)
                break

    return tokens

abbre_last = ['s','re','ve','t','m','ll']
abbre_sym = ["'","?","’"]
################################################################################
unseen_words = ['hiagnnamalnsw']

def findhashtag_offset(query_tokens):
    bool_token = False
    found_offsets = []
    first_offset = 0
    end_offset = 0
    for i, token in enumerate(query_tokens):
        if token[0] == '#':
    
            if bool_token:
                found_offsets.append([first_offset, end_offset])
            first_offset = token[1]
            end_offset = token[2]
            bool_token = True
        else:
            if bool_token:
                if end_offset == token[1] -1:
                    end_offset = token[2]
                    if i==len(query_tokens)-1:
                        found_offsets.append([first_offset, end_offset])
                else:
                    found_offsets.append([first_offset, end_offset])
                    bool_token = False
    return found_offsets

def replace_words(sub_output,sub_off_set,return_query_tokens):
    one_d_sub_off = []
    for off in sub_off_set:
        one_d_sub_off.extend(off)
    one_d_sen = []
    for token in sub_output:
        one_d_sen.extend(token)
        
    new_return= []
    for subsuboff in return_query_tokens:
        bool_matched = False
        for i, s in enumerate(one_d_sub_off):
            if s[0] >= subsuboff[1] and s[0] <= subsuboff[2] and \
            s[1] >= subsuboff[1] and s[1] <= subsuboff[2]:
                bool_matched = True
                new_return.append(tuple([one_d_sen[i], s[0], s[1]]))
        if not bool_matched:
            new_return.append(subsuboff)
    return new_return

def extract_sim(tweet, keys, full=0, abb_dict={}):
    '''Extracts all location names from a tweet.'''
#    abb_dict = {}
#    abb_dict['u.s.'] = 'us'
#    abb_dict['u.s.a'] = 'usa'

    # --------------------------------------------------------------------------

    #will contain for example: (0, 11): [(u'new avadi road', 3)]
#    print(ord(tweet[6]))
    
#    nfkd = unicodedata.normalize('NFKD', tweet)
#    nfkd.replace("’", "'").encode('utf-8')
    new_tweet = ""
#    pdb.set_trace()
    for char in tweet:
        if ord(char)==8217:
            new_tweet = new_tweet+"'"
        else:
            new_tweet = new_tweet+char
    tweet = new_tweet
#    print(new_tweet)
#    return str(nfkd.encode('ASCII', 'ignore').decode('ASCII'))
#
#    valid_ngrams = defaultdict(list)
#    print()
    
#    tweet.replace("’", "'").encode('utf-8')
#    pdb.set_trace()
    tweet = strip_non_ascii(tweet)

    # we will call a tweet from now onwards a query
    query = tweet; str(tweet.lower())
    remove_men = 0
    preprocessed_query = preprocess_tweet(query,keys, remove_men)
    if remove_men:
#    print(preprocessed_query)
        query_tokens = align_and_split(query, preprocessed_query)
    else:
        new_preprocessed_query = re.sub(mentions_re, " @ ", preprocessed_query)
        query_tokens = align_and_split(query, new_preprocessed_query)
        ori_query_tokens = align_and_split(query, preprocessed_query)
#    print(query_tokens)
#    pdb.set_trace()

    if full:
        return_query_tokens = copy.deepcopy(ori_query_tokens)
        
    hashtag_offsets = findhashtag_offset(query_tokens)
#    pdb.set_trace()

    #print(query_tokens)
    # --------------------------------------------------------------------------
    # prune the tree of locations based on the exisitence of stop words
    # by splitting the query into multiple queries
#    query_splits = Twokenize.tokenize(query)
    #stop_in_query =  set(query_splits)

    # if tweet contains the following then remove them from the tweet and
    # split based on their presence: stop_in_query | '.',
    extra_stop = set(['?']) #".", 
    comma_stop_exc = ['st','t','T','ST','St','s','n','e','w','se', 'ne']

    stop_in_query =set(
        ['[', ']','/',',','@','&','#','$','%', '(', ')', '.', '|', '!', ';', ':', '<', '>', "newline"]) #"#", 
    # remove stops from query
    stop_index = []
    del_index = []
    insert_index = []
    insert_tuple = []
    token_len = len(query_tokens)
    for index, token in enumerate(query_tokens):
        if token[0] in stop_in_query:
           if not (token[0] == '.' and query_tokens[index-1][0] in comma_stop_exc):
               stop_index.append(index)
            #query_tokens[index] = ()
        if index > 0 and token[0] in extra_stop and  (query_tokens[index][1]-query_tokens[index-1][2]) >1:
            stop_index.append(index)
        if index < token_len-1 and token[0] in extra_stop and  (query_tokens[index+1][1]-query_tokens[index][2]) >1:
            stop_index.append(index)
        if index > 0 and index < token_len-1 and token[0] == '?' \
                 and query_tokens[index+1][0] in abbre_last \
                 and query_tokens[index-1][2]+1 == query_tokens[index][1] \
                 and query_tokens[index][2]+1 == query_tokens[index+1][1]:

            del_index.append(index-1)
            del_index.append(index)
            del_index.append(index+1)
            insert_index.append(index-1-2*len(insert_tuple))
            new_tuple = tuple([query_tokens[index-1][0] + "'" + query_tokens[index+1][0],query_tokens[index-1][1], query_tokens[index+1][2]])
            insert_tuple.append(new_tuple)
    stop_index =list(set(stop_index))
#    pdb.set_trace()

    for index in stop_index:
        query_tokens[index] = ()
    query_tokens = [query_tokens[i] for i in range(token_len) if i not in del_index ]
    for i, index in enumerate(insert_index):
        query_tokens.insert(index,insert_tuple[i])
    # combine consecutive tokens in a query as possible location names
    query_filtered = list()
    candidate_location = {"tokens": list(), "offsets": list()}
#    pdb.set_trace()

    for index, token in enumerate(query_tokens):
        if len(token) > 0:
            candidate_location["tokens"].append(token[0].strip())
            candidate_location["offsets"].append((token[1], token[2]))

        elif candidate_location != "":
            query_filtered.append(candidate_location)
            candidate_location = {"tokens": list(), "offsets": list()}

        # if I am at the last token in the list
        # ex: "new avadi road" then it wont be added unless I know that this is
        #        the last token then append whatever you have
        if index == len(query_tokens) - 1:
            query_filtered.append(candidate_location)
    # Remove empty query tokens
    sub_output = []
    sub_off_set = []
    query_tokens = [qt[0] for qt in query_tokens if (qt != tuple())]
    #remove non special characters
#    pdb.set_trace()
#    pdb.set_trace()

    for sub_query in query_filtered: # ----------------------------------- for I
        sub_query_tokens = sub_query["tokens"]
        if len(sub_query_tokens) == 0:
            continue
        new_tokens = []
        new_offsets = []
        sub_offset = sub_query["offsets"]
        for c_idx, i in enumerate(sub_query_tokens):
            
#            if i == 'U.S.':
#                pdb.set_trace()
#            pdb.set_trace()

            if i.lower() not in abb_dict.keys():
                temp = re.split('[^a-zA-Z^0-9]',i)
            else:
                new_to = ''
                for char in i:
                    if char != '.':
                        new_to += char
                temp = [new_to]
            offset = sub_offset[c_idx]
            split_count = 0
            sym_match = 0
            for s in abbre_sym:
                if s in i:
                    sym_match = 1
                    break
            if sym_match and len(temp)>= 2 and temp[-1] in abbre_last:
                if temp[-1] == 'm':
                    temp[-1] = 'am'
                if temp[-1] == 't':
                    temp[-1] = 'not'
                if temp[-1] == 've':
                    temp[-1] = 'have'
                if temp[-1] == 're':
                    temp[-1] = 'are'
                if temp[-1] == 'll':
                    temp[-1] = 'will'
                if temp[-1] == 's':
                    del temp[-1]

            for t in temp:
                if t:
                    if hasNumbers(t):

                        groups = re.split('(\d+)',t)
                        groups = [g for g in groups if g]
                        start_index = offset[0]
                        for g in groups:
#                            if g and hasNumbers(g):
#                                num_l = len(g)
#                                new_g = '0'*num_l
#                                new_tokens.append(new_g)
#                                split_count+=1
#                            else:
                            if g:
                                new_tokens.append(g)
                                split_count+=1
                                if len(temp)==1:
                                    new_offsets.append((start_index, start_index+len(g)-1))
                                    start_index=start_index+len(g)
                                else:
                                    new_offsets.append(offset)

                    else:
                        new_tokens.append(t)
                        split_count+=1
                        new_offsets.append(offset)

#            for KT in range(split_count):
#                new_offsets.append(offset)
        new_output = []
        new_offset_out = []
        for pp, w in enumerate(new_tokens):
            if w.lower() not in keys and not hasNumbers(w):
                # when a word is unknow, first try to segment the word. Otherwise correct it
#                segments = segment(w)
#                start_p = 0
#                for i, seg in enumerate(segments):
#                    segments[i]=w[start_p:start_p+len(seg)]
#                    start_p=start_p+len(seg)
#                if len(segments) > 1:
#                    for seg in segments:
#                        new_output.append(seg)
#                        new_offset_out.append(new_offsets[pp])
#                #correct the word
#                else:
                new_w = w
#                new_w = spell.correction(w)
#                new_w = new_w.replace(" ", "")
#                if (new_w == w):
#                    new_w = unseen_words[0]
                new_output.append(new_w)
                new_offset_out.append(new_offsets[pp])
            else:
                new_output.append(w)
                new_offset_out.append(new_offsets[pp])
        if new_output:
            sub_output.append(new_output)
            sub_off_set.append(new_offset_out)
    if full:
        return_query_tokens = replace_words(sub_output,sub_off_set,return_query_tokens)
        return sub_output,sub_off_set,return_query_tokens, hashtag_offsets
    else:
        return sub_output,sub_off_set

if __name__== "__main__":
#    print(extract_sim('@CNBC_Awaaz Magma Fincrop Ltd. buy 3th @166.25 hold or sell?', [], full=1))
#    print(extract_sim('RT @HTownRD: #HoustonFlood at 34th & White Oak Bayou https://t.co/EQiY06SkQd', [], full=1))
    file_name = 'data/osm_abbreviations_globe.csv'
        
    sim_abv = abbrevison1(file_name)
    abv_punk = {}
    for key in sim_abv.keys():
        new_abv = ''
        for i, char in enumerate(key):
#            if key == 'us':
#                pdb.set_trace()
#            if i != len(key)-1:
                new_abv += char + '.'
#            else:
#                new_abv += char
        abv_punk[new_abv]=key
    print(abv_punk)
#    sub_output,sub_off_set,return_query_tokens, hashtag_offsets = extract_sim("RT @Nervana_1: Welcome to the Southern province of #Iran RT @AlArabiya_Eng: Houthi rebels seize control of main entrances",[], 1, abv_punk)

    sub_output,sub_off_set,return_query_tokens, hashtag_offsets = extract_sim("Read the latest #Turkey Crackdown Chronicle , CPJ ’s weekly round-up of #pressfreedom violations in the #country",[], 1, abv_punk)
    print(return_query_tokens, hashtag_offsets)