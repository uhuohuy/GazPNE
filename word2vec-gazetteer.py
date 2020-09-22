#word2vec
from gensim.models import Word2Vec
import argparse

def load_garzeteer_data(pos_f):
    training_data = []
    max_len = 0
    with open(pos_f, 'r',encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            temp=[]
            tokens = line.split(' ')
            for tok in tokens:
               temp.append(str(tok))
            training_data.append(temp)
            if max_len < len(temp):
               max_len = len(temp)
    return training_data, max_len

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--data', type=int, default=146)
    parser.add_argument('--osmembed', type=float, default=2)
    args = parser.parse_args()
    print ('data: '+str(args.data))
    print ('osmembed: '+str(args.osmembed))
    model_file = "osm.model"
    
    pos_f = 'data/positive'+str(args.data)+'.txt'
    osm_save = 'data/osm_vector'+str(args.osmembed)+'.txt'
    osm_char_save = 'data/osm_char_vector'+str(args.osmembed)+'.txt'

    # word embedding training
    sentence,max_len = load_garzeteer_data(pos_f)
    model = Word2Vec(size=30, window=2, min_count=1)
    model.build_vocab(sentence)
    model.train(sentences=sentence, total_examples = model.corpus_count, epochs = 50)
    model.wv.save_word2vec_format(osm_save,binary = False)
    print('word trained')
    


