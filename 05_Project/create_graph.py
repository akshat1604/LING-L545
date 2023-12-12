import os
import sys
import spacy
import pandas as pd
import numpy as np
from itertools import combinations
from transformers import AutoTokenizer, AutoModelForTokenClassification
from preprocess import preprocess_strings
from ner import main as ner
import parser

sys.path.append('..')

config = parser.get_config()

config.early_stop = True
config.dataset = "i2b2"
config.border_size = -1
config.use_test = True

post = '_' + config.preprocessing_type
config.data_root = "pre-processed/" + config.preprocessing_type + "/"

config.embedding_file = 'wikipedia-pubmed-and-PMC-w2v.txt'
config.test_text_dataset_path = 'test_new.txt'

def return_index_of_sublist(words,query_list):
    for idx in range(len(words) - len(query_list) + 1):
        if words[idx: idx + len(query_list)] == query_list:
            return idx
        
def cut_string_by_words(string, num_words):
    words = string.split()
    cut_words = words[:num_words]
    return ' '.join(cut_words)

def main(df,index, tokenizer, model):

    ## Load csv

    ## Get one sentence
    preprocessed_doc = df["TEXT"].iloc[index]

    ## preprocess using medspacy
    # preprocessed_doc = str(preprocess_strings(text)).lower()

    ## perform ner
    multi_word_entities = list(set(ner(preprocessed_doc, tokenizer, model)))
    
    ## split the sentences to have sentence numbers
    ent_sent_map = {}
    final_sentence_list = []
    colon_split = preprocessed_doc.split(':')

    for sent in colon_split:
        new_list = sent.strip().split('.')
        for new in new_list:
            if len(new.strip().split()) > 4:
                final_sentence_list.append(new)
        # final_sentence_list.extend(new_list)
    
    ## adding values to map
    for ents in multi_word_entities:
        for idx, sent in enumerate(final_sentence_list):
            if sent.find(ents) != -1:
                ent_sent_map[ents] = idx
    
    word_combinations = combinations(multi_word_entities, 2)

    text_file = open("actual_test.txt", "w")

    for pair in word_combinations:
        first, second = pair

        sent_id1 = ent_sent_map.get(first)
        sent_id2 = ent_sent_map.get(second)
        
        start1, end1, start2, end2 = None, None, None, None

        if sent_id1 is None or sent_id2 is None:
            continue

        if sent_id1 == sent_id2:

            sentence = final_sentence_list[sent_id1]
                
            sent_before = final_sentence_list[sent_id1-1] if sent_id1 > 0 else None
            sent_after = final_sentence_list[sent_id1+1] if sent_id1+1 < len(final_sentence_list) else None
            if sent_before is not None:
                sentence = sent_before + ". " + sentence
        
            if sent_after is not None:
                sentence = sentence + ". " + sent_after

            sentence = cut_string_by_words(sentence,190)
            
            words = sentence.replace(".","").split()
            first_ent_split = first.split()
            second_ent_split = second.split()

            start1 = return_index_of_sublist(words,first_ent_split)
            end1 = start1 + len(first_ent_split) - 1 if start1 is not None else None

            start2 = return_index_of_sublist(words,second_ent_split)
            end2 = start2 + len(second_ent_split) - 1 if start2 is not None else None

            string_to_write = f"{0} {start1} {end1} {start2} {end2} {sentence}\n"

            if start1 is not None and start2 is not None:
                text_file.write(string_to_write)
            
        elif sent_id2 -  sent_id1 < 3:
            sentence1 = final_sentence_list[sent_id1]
            sentence2 = final_sentence_list[sent_id2]
            sentence = sentence1 + ". " + sentence2
                
            sent_before1 = final_sentence_list[sent_id1-1] if sent_id1 > 0 else None
            sent_before2 = final_sentence_list[sent_id2-1] if sent_id2 > 0 else None

            sent_after1 = final_sentence_list[sent_id1+1] if sent_id1+1 < len(final_sentence_list) else None
            sent_after2 = final_sentence_list[sent_id2+1] if sent_id2+1 < len(final_sentence_list) else None

            if sent_before1 is not None:
                sentence = sent_before1 + ". " + sentence

            if sent_after1 is not None:
                sentence = sentence + ". " + sent_after1

            if sent_before2 is not None:
                sentence = sentence + ". " + sent_before2
            
            if sent_after2 is not None:
                sentence = sentence + ". " + sent_after2
            
            sentence = cut_string_by_words(sentence,190)
            words = sentence.replace(".","").split()
            first_ent_split = first.split()
            second_ent_split = second.split()

            start1 = return_index_of_sublist(words,first_ent_split)
            end1 = start1 + len(first_ent_split) - 1 if start1 is not None else None

            start2 = return_index_of_sublist(words,second_ent_split)
            end2 = start2 + len(second_ent_split) - 1 if start2 is not None else None

            string_to_write = f"{0} {start1} {end1} {start2} {end2} {sentence}\n"
            
            if start1 is not None and start2 is not None:
                text_file.write(string_to_write)

    
    text_file.close()

    return final_sentence_list, ent_sent_map


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("sschet/bert-base-uncased_clinical-ner")
    ner_model = AutoModelForTokenClassification.from_pretrained("sschet/bert-base-uncased_clinical-ner")

    main("sample.csv",0,tokenizer,ner_model)