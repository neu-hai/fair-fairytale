import pandas as pd
from src.utils import read_story_txt_file, get_tfidf_model, get_tfidf_dict, get_tfidf_score
from nltk.stem.porter import PorterStemmer
import nltk



def tftokenize(token):
    token = nltk.word_tokenize(token)
    return PorterStemmer().stem(token[0])


def get_named_dict(story_text):
    path = './data/FairytaleQA/story_txts'
    tfidf,story_dict = get_tfidf_model(path, story_text)
    return get_tfidf_dict(tfidf, story_text)
def append_major(actions_df, name_dict):
    major = {}
    for idx in range(actions_df.shape[0]):
        row = actions_df.iloc[idx]
        s_id, v_id = row['sentence_id'], row['verb_id']

        verb = tftokenize(row['verb'])

        if s_id not in major:
            try:
                major[s_id] = (v_id,name_dict[verb])
                # assign low score for those functional phrase
                if verb in ['said','say']:
                    major[s_id] = (v_id,0)
            except:
                major[s_id] = (v_id,0)
        else:
            try:
                if name_dict[verb]>major[s_id][1]:
                    major[s_id] = (v_id,name_dict[verb])
                    # assign low score for those functional phrase
                    if verb in ['said', 'say']:
                        major[s_id] = (v_id, 0)
            except:
                pass
    majorls = []
    for idx in range(actions_df.shape[0]):
        row = actions_df.iloc[idx]
        s_id, v_id = row['sentence_id'], row['verb_id']
        if v_id==major[s_id][0]:
            majorls.append(True)
        else:
            majorls.append(False)
    actions_df['major_event'] = majorls
    return actions_df

def get_major_actions(actions_df,story_text):
    name_dict = get_named_dict(story_text)
    actions_df = append_major(actions_df, name_dict)
    return actions_df[actions_df['major_event']==True]

