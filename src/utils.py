import csv
import os
import re
import string

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

def convert_story_csv_to_text(story_csv_file: str) -> str:
    story_text = ""
    with open(story_csv_file, newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            text = row['text'] + ' '
            story_text += text
    story_text = story_text[:-1]
    
    story_text = re.sub('\n\n', ' ', story_text)
    story_text = re.sub('\r\r', ' ', story_text)
    story_text = re.sub('\n', ' ', story_text)
    story_text = re.sub('\r', ' ', story_text)
    
    return story_text

def convert_story_csv_to_txt_file(story_csv_file: str, story_txt_file: str):
    story_text = ""
    with open(story_csv_file, newline='') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            text = row['text'] + ' '
            story_text += text
    story_text = story_text[:-1]
    
    story_text = re.sub('\n\n', ' ', story_text)
    story_text = re.sub('\r\r', ' ', story_text)
    story_text = re.sub('\n', ' ', story_text)
    story_text = re.sub('\r', ' ', story_text)
    
    with open(story_txt_file, "w") as txtfile:
        txtfile.write(story_text)
    
    print("Story txt file saved at: {}".format(story_txt_file))

def read_story_txt_file(story_txt_file: str) -> str:
    with open(story_txt_file) as txt_file:
        story_text = txt_file.read()
    return story_text

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

def get_tfidf_model(path,current_text=None):
    token_dict = {}
    for dirpath, dirs, files in os.walk(path):
        for f in files:
            fname = os.path.join(dirpath, f)
            with open(fname) as pearl:
                text = pearl.read()
                token_dict[f] = text.lower().translate(str.maketrans('','',string.punctuation))
    if current_text:
        token_dict['current'] = current_text
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    tfs = tfidf.fit_transform(token_dict.values())
    return tfidf,token_dict

def get_tfidf_dict(tfidf, story:str):
    response = tfidf.transform([story])
    feature_names = tfidf.get_feature_names_out()
    names_to_score = {}
    for col in response.nonzero()[1]:
        names_to_score[feature_names[col]] = response[0, col]
    return names_to_score

def get_tfidf_score(name_dict, token):
    token = tokenize(token)[0]
    return name_dict[token]

def get_evls(df):
    evls = []
    last_sid = -1
    for i in range(df.shape[0]):
        row = df.iloc[i]
        if row['sentence_id']==last_sid:
            evls[-1]+= (', '+row['verb'])
        else:
            evls.append(row['verb'])
            last_sid = row['sentence_id']
    return evls
    
def get_events_df(major, evls, sents):
    new = major[['sentence_id','verb_id','verb']]
    new['verb_ls'] = evls
    s_ids = new['sentence_id'].to_list()
    sents = [s for i,s in enumerate(sents) if i in s_ids]
    # sent = sent.loc[sent['sentence_id'].isin(new['sentence_id'].to_list())]
    new['sentence'] = sents
    return new
