import os
import re

import pandas as pd

from src.sentences import split_sentences

def read_story_txt_file(story_txt_file: str) -> str:
    with open(story_txt_file) as txt_file:
        story_text = txt_file.read()
    return story_text

def get_start_end_bytes_of_sentences(text):
    sentences = split_sentences(text)
    
    while ('' in sentences):
        sentences.remove('')
                
    start_end_sentences = []
    for sentence in sentences:
        sentence_start = text.find(sentence)
        sentence_end = sentence_start + len(sentence) + 1
        start_end_sentences.append([text[sentence_start:sentence_end], sentence_start, sentence_end])
        
    return start_end_sentences

data_dir = '/Users/pti/challenges/4761_fair_fairytale/fair-fairytale-nlp/data/FairytaleQA/split_by_origin/'

origin_dirs = os.listdir(data_dir)

if '.DS_Store' in origin_dirs:
    origin_dirs.remove('.DS_Store')

print(origin_dirs)

for origin_dir in origin_dirs:
    story_dir = data_dir + origin_dir
    story_files = [f for f in os.listdir(story_dir) if f[-3:] == 'txt']
    for story_file in story_files:
        story_id = story_file[:-4]
        story_text = read_story_txt_file(story_dir + '/' + story_file)
        start_end_sentences = get_start_end_bytes_of_sentences(story_text)
        sentences_df = pd.DataFrame(start_end_sentences).reset_index().rename(columns = {'index': 'sentence_id', 0: 'text', 1: 'start', 2: 'end'})
        output_sentences_file = os.path.join('/Users/pti/challenges/4761_fair_fairytale/fair-fairytale-nlp/data/pipeline/sentences/', story_id + '.sentences.csv')
        print(output_sentences_file)
        sentences_df.to_csv(output_sentences_file, index = False)
