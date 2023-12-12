import numpy as np
import random
from spacy.lang.en.stop_words import STOP_WORDS as stop_words
from collections import Counter

def get_data(res, dataset, config, mode='normal'):
    def transpose(data):
        # convert the list of tuples (outer dimension num of sentences) to tuple of lists (outer dim features)
        transposed_data = []
        for x in zip(*data):
            transposed_data.append(list(x))
        return tuple(transposed_data)

    test_text_dataset_file = res(config.test_text_dataset_path)


    if config.cross_validate is False and config.cross_validate_report is False:

        dev_data = open(test_text_dataset_file, 'r') # means we will report test scores
        dev_data = preprocess_data_noncrossvalidated(dev_data, config.border_size)


    return dev_data, test_text_dataset_file


def split_data_cut_sentence(data, border_size=-1):
    sentences = []
    relations = []
    e1_pos = []
    e2_pos = []
    # is_reversed = []

    # In the parsed data: Num1 num2 num3 num4 num5 sentence
    # Num1 - relation number
    # Num2 - left entity start (starts the numbering from 0)
    # Num3 - left entity end
    # Num4 - right entity start
    # Num5 - right entity end
    if border_size < 0:
        for line in data:
            line = line.strip().lower().split()
            left_start_pos = int(line[1])
            right_end_pos = int(line[4])
            relations.append(int(line[0]))
            e1_pos.append( (int(line[1]), int(line[2])) ) # (start_pos, end_pos)
            e2_pos.append( (int(line[3]), int(line[4])) ) # (start_pos, end_pos)
            # is_reversed.append( float(isreversed_dictionary[int(line[0])]) )
            sentences.append(line[5:])
    else:
        for line in data:
            line = line.strip().lower().split()
            left_start_pos = int(line[1])
            right_end_pos = int(line[4])
            if left_start_pos < right_end_pos:
                relations.append(int(line[0]))
                # is_reversed.append( float(isreversed_dictionary[int(line[0])]) )
                sentence = line[5:]
                len_sen = len(sentence)
                if left_start_pos >= border_size:
                    left_border_size = border_size
                else:
                    left_border_size = left_start_pos
                e1_pos.append( (left_border_size, int(line[2])-left_start_pos+left_border_size) ) # (start_pos, end_pos)
                e2_pos.append((int(line[3])-left_start_pos+left_border_size, int(line[4])-left_start_pos+left_border_size)) # (start_pos, end_pos)
                sentences.append(sentence[(left_start_pos-left_border_size):min(right_end_pos+border_size+1, len_sen)])

    return sentences, relations, e1_pos, e2_pos

def openFileAsList(filename):
    with open(filename) as f:
        mylist = [line.rstrip('\n') for line in f]
    return mylist

def preprocess_data_noncrossvalidated(data, border_size):
    data = split_data_cut_sentence(data, border_size)
    return data


def build_dict(sentences, low_freq_thresh=0, remove_stop_words=False):
    word_count = Counter()
    for sent in sentences:
        if sent is not None:
            for w in sent:
                if remove_stop_words is True and w in stop_words:
                    # first make sure to put stop words at the end so that they don't leave
                    # holes in the indexing
                    word_count[w] = -1
                    # make sure that they come after the words with frequency 1
                else:
                    word_count[w] += 1

    # the words from the low_freq_thresh wouldn't leave holes in the indexing because every word with
    # an index higher than them will be mapped to 0
    ls = word_count.most_common()
    # above organizes the words by most common and less common words; at this point we have the counts

    dictionary = {}
    for index, word_and_count in enumerate(ls):
        word = word_and_count[0]
        if remove_stop_words is True and word in stop_words:
            dictionary[word] = 0 #giving it a zero index
        elif low_freq_thresh > 0 and word_count[word] <= low_freq_thresh:
            dictionary[word] = 0
        else:
            dictionary[word] = index + 1
    return dictionary

def get_word_dict(data, low_freq_thresh, early_stop):
    # Build vocab, pretend that your test set does not exist because when you need to use test 
    # set, you can just make sure that what we report on (i.e. dev set here) is actually the test data
    all_data = data['dev'][0]

    word_dict = build_dict(all_data, low_freq_thresh)
    return word_dict

def vectorize(config, data, word_dict):
    def assign_splits(pos1, pos2):
        if pos1[1] < pos2[1]:
            return pos1[1], pos2[1]
        elif pos1[1] > pos2[1]:
            return pos2[1], pos1[1]
        elif config.use_piecewise_pool is True and config.dataset == 'i2b2':
            if pos1[0] < pos2[0]: return pos1[0], pos2[0]
            elif pos1[0] > pos2[0]: return pos2[0], pos2[1]
            else: raise Exception("Both entities overlap exactly")
        elif config.use_piecewise_pool is True:
            raise Exception("Entity positions cannot end at the same position for piecewise splitting")
            # I anticipate the above to be a problem for NER blinding, where there are 
            # overlaps between the entity pairs because the existence of the NER label extends the
            # entity pairs
        else:
            return pos1[1], pos2[1] # this is not going to be used anyway, but using these is problematic

    sentences, relations, e1_pos, e2_pos = data
    
    max_sen_len = config.max_len
    max_e1_len = config.max_e1_len
    max_e2_len = config.max_e2_len
    num_data = len(sentences)
    local_max_e1_len = max(list(map(lambda x: x[1]-x[0]+1, e1_pos)))
    local_max_e2_len = max(list(map(lambda x: x[1]-x[0]+1, e2_pos)))
    print('max sen len: {}, local max e1 len: {}, local max e2 len: {}'.format(max_sen_len, local_max_e1_len, local_max_e2_len))

    # maximum values needed to decide the dimensionality of the vector
    sents_vec = np.zeros((num_data, max_sen_len), dtype=int)
    e1_vec = np.zeros((num_data, max_e1_len), dtype=int)
    e2_vec = np.zeros((num_data, max_e2_len), dtype=int)
    # dist1 and dist2 are defined in the compute distance function
    
    position1 = [] # need to populate this way because want to make sure that the splits are in order
    position2 = []
    for idx, (sent, pos1, pos2) in enumerate(zip(sentences, e1_pos, e2_pos)):
        # all unseen words are mapped to the index 0
        vec = [word_dict[w] if w in word_dict else 0 for w in sent]
        sents_vec[idx, :len(vec)] = vec
        
        split1, split2 = assign_splits(pos1, pos2)
        position1.append(split1)
        position2.append(split2)

        # for the particular sentence marked by idx, set the entry as the vector gotten from above
        # which is basically just a list of the indexes of the words
        for ii in range(max_e1_len):
            if ii < (pos1[1]-pos1[0]+1):
                    e1_vec[idx, ii] = vec[range(pos1[0], pos1[1]+1)[ii]]
                    # this is assigning the particular sentence's e1 val to have the index of the corresponding word
            else:
                    e1_vec[idx, ii] = vec[pos1[-1]]
                    # in the above case it is grabbing the last word in the entity and padding with that

        for ii in range(max_e2_len):
            if ii < (pos2[1]-pos2[0]+1):
                    e2_vec[idx, ii] = vec[range(pos2[0], pos2[1]+1)[ii]]
            else:
                    e2_vec[idx, ii] = vec[pos2[-1]]

    dist1, dist2, num_pos = relative_distance(num_data, max_sen_len, e1_pos, e2_pos)

    return sents_vec, np.array(relations).astype(np.int64), e1_vec, e2_vec, dist1, dist2, position1, position2

def relative_distance(num_data, max_sen_len, e1_pos, e2_pos):
    dist1 = np.zeros((num_data, max_sen_len), dtype=int)
    dist2 = np.zeros((num_data, max_sen_len), dtype=int)
    # compute relative distance
    #TODO: (geeticka) think about what to do for the cases when e1_pos and e2_pos is None
    for sent_idx in range(num_data):
        for word_idx in range(max_sen_len):
            if e1_pos[sent_idx] is None or e2_pos[sent_idx] is None:
                continue
            if word_idx < e1_pos[sent_idx][0]:
                    dist1[sent_idx, word_idx] = pos(e1_pos[sent_idx][0] - word_idx)
            # in the above the word is behind the e1's word
            elif word_idx > e1_pos[sent_idx][1]:
                    dist1[sent_idx, word_idx] = pos(e1_pos[sent_idx][1] - word_idx)
            # the word is after the e1
            else:
                    dist1[sent_idx, word_idx] = pos(0)
            # the word is within the entity

            if word_idx < e2_pos[sent_idx][0]:
                    dist2[sent_idx, word_idx] = pos(e2_pos[sent_idx][0] - word_idx)
            elif word_idx > e2_pos[sent_idx][1]:
                    dist2[sent_idx, word_idx] = pos(e2_pos[sent_idx][1] - word_idx)
            else:
                    dist2[sent_idx, word_idx] = pos(0)

    num_pos = max(np.amax(dist1), np.amax(dist2)) - min(np.amin(dist1), np.amin(dist2))
    return dist1, dist2, num_pos

def pos(x):
        '''
        map the relative distance between [0, 123)
        '''
        if x < -60:
                        return 0
        if x >= -60 and x <= 60:
                        return x + 61
        if x > 60:
                        return 122