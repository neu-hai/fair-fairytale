import json
import re
import pandas as pd
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer as twt
import spacy
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from allennlp.predictors.predictor import Predictor
from os import path
import os
from allennlp_models import pretrained
import pickle

nlp = spacy.load("en_core_web_sm")
def spacy_span(sent):
    """
    get a list of spans using spacy tokenizer
    """
    spans = []
    #     sent = sentences[sent_id]
    doc = nlp(sent)
    token_id = 0
    for offset in range(len(sent)):
        if token_id >= len(doc):
            continue
        token=doc[token_id]

        if sent[offset: offset+len(token)] == token.text:
            spans.append((offset, offset+len(token)))
            token_id+=1
    #             print(sent[offset: offset+len(token)],token.text)
    return spans

def spacy_tokenize(sent):
    """
    output a list of tokens, on par with ntlk word_tokenize
    """
    #     sent = sentences[sent_id]
    doc = nlp(sent)
    out = []
    for token in doc:
        out.append(token.text)
    return out



def byteParse(tokens, sentence):
    """
    parse a sentence into ordered dict, needed by TE data format
    """
    token_start_end_bytes = get_tokens_start_end_bytes(tokens, sentence)
    pos_tags_sentence = OrderedDict()
    for i in range(len(tokens)):
        try:
            span = (token_start_end_bytes[i]['start_byte'],token_start_end_bytes[i]['end_byte'])
        except:
            print("error happens, at", i, "and the dict is", token_start_end_bytes[i])
        pos_tags_sentence['['+str(span[0])+':'+ str(span[1]) + ')'] = (token_start_end_bytes[i]['token'], 'NOUN')
    return pos_tags_sentence

def get_tokens_start_end_bytes(tokens, sentence):
    token_start_end_bytes_sentence = []

    s = 0
    for i, token in enumerate(tokens):
        token_start_end_bytes = {}

        #print("token: ", token)
        token_start_end_bytes['token'] = token
        token_start_end_bytes['i'] = i

        start_byte = s
        end_byte = s + len(token)

        while sentence[start_byte] == ' ':
            start_byte += 1
            end_byte += 1

        #print(start_byte, end_byte)
        #print(sentence[start_byte])
        #print(sentence[end_byte])
        #print("mapping :", token, sentence[start_byte:end_byte])
        #print("\n")

        if token != sentence[start_byte:end_byte]:
            print('Incorrect mapping of token to sentence bytes.')
            print(token, sentence[start_byte:end_byte])
            print(sentence,'\n')
            print(tokens)
            return 'Incorrect mapping of token to sentence bytes.'
            #search for the next byte

        token_start_end_bytes['start_byte'] = start_byte
        token_start_end_bytes['end_byte'] = end_byte

        token_start_end_bytes_sentence.append(token_start_end_bytes)

        s = end_byte

    return token_start_end_bytes_sentence

def event_span(sent_id, verb_id, srl, sentences):
    """
    need srl, sentences as a preset global variable
    """
    if 'B-V' in srl[sent_id]['verbs'][verb_id]['tags']:
        idx = srl[sent_id]['verbs'][verb_id]['tags'].index('B-V')
    else:
        tags = srl[sent_id]['verbs'][verb_id]['tags']
        tags = ['B-ARGM' if 'B-ARGM' in x else x for x in tags ]
        idx = tags.index('B-ARGM')
    #     sent_token_span = list(twt().span_tokenize(sentences[sent_id]))
    token_start_end_bytes_test = get_tokens_start_end_bytes(srl[sent_id]['words'], sentences[sent_id])
    s,e = token_start_end_bytes_test[idx]['start_byte'], token_start_end_bytes_test[idx]['end_byte']-1
    return (s,e)

class Event():
    def __init__(self, id, type, text, tense, polarity, span):
        self.id = id
        self.type = type
        self.text = text
        self.tense = tense
        self.polarity = polarity
        self.span = span

def create_data_instance(sent_id1, sent_id2, verb_id1, verb_id2, sentences, srl, event_coref):
    """
        sent_id1, and sent_id2 are two sentences we want to run
        verb_id1, is the verb_id in sent1, verb_id2, is the verb_id in sentence2


    :param sent_id1:
    :param sent_id2:
    :param verb_id1:
    :param verb_id2:
    :param sentences:
    :param srl:
    :param event_coref:
    :return:  one single data instance required by ECONET TE_orderin input format
    """
    e1_span = event_span(sent_id1, verb_id1,srl, sentences)
    e2_span = event_span(sent_id2, verb_id2,srl, sentences)

    # write the assertion code to make sure our calculated span matches, Paulina's span; could improve efficiency
    row1 = event_coref.loc[(event_coref['sentence_id'] == sent_id1) & (event_coref['verb_id'] == verb_id1)]
    row2 = event_coref.loc[(event_coref['sentence_id'] == sent_id2) & (event_coref['verb_id'] == verb_id2)]
    if row1.shape[0]:

        assert (row1.iloc[0]['verb_start_byte'],row1.iloc[0]['verb_end_byte']-1) == e1_span
    if row2.shape[0]:

        assert (row2.iloc[0]['verb_start_byte'],row2.iloc[0]['verb_end_byte']-1) == e2_span

    if sent_id1 != sent_id2:
        combo_sent = sentences[sent_id1] +' '+ sentences[sent_id2]
        combo_tokens =srl[sent_id1]['words'] + srl[sent_id2]['words']
        doc_dict = byteParse(combo_tokens,combo_sent)

        # since sentence1 is appended in front of sentence2, need to add offset value to e2_span
        offset = len(sentences[sent_id1])+1
        e2_span =  (e2_span[0] + offset, e2_span[1]+offset)
    else:
        combo_sent = sentences[sent_id1]
        combo_tokens = srl[sent_id1]['words']
        doc_dict = byteParse(combo_tokens,combo_sent)


    if combo_sent[e2_span[0]: e2_span[1]+1] != srl[sent_id2]['verbs'][verb_id2]['verb']:
        print("currently processing the following event pair")
        print(sent_id1, sent_id2, verb_id1, verb_id2)
        print(combo_sent[e2_span[0]: e2_span[1]+1], srl[sent_id2]['verbs'][verb_id2]['verb'])
        print(e2_span)
        print(sent_id2)

    assert combo_sent[e1_span[0]: e1_span[1]+1] == srl[sent_id1]['verbs'][verb_id1]['verb']
    assert combo_sent[e2_span[0]: e2_span[1]+1] == srl[sent_id2]['verbs'][verb_id2]['verb']

    # below stay the same; the other parameters of EVENT are not required, so we just
    # use NONE type as place holder
    left_event = Event(None, None, None, None, None, e1_span)
    right_event = Event(None, None, None, None, None, e2_span)
    v = dict()
    v['rel_type'],v['rev'],v['doc_dictionary'],v['event_labels'],v['doc_id'],v['left_event'],v['right_event'] = \
        'BEFORE', None, doc_dict, None, None, left_event, right_event
    return v


def create_data_instances_multi_events(TE_ids,sentences, srl, event_coref):
    """
    Main function to create data create_data_instances that can be stored in pickle files
    It takes a TE_ids file, eg:  list [ list [s_id,s_id2,e_id,e_id2],  [s_id',s_id2',e_id',e_id2'], ...    ]
    Rather than extracting one event per sentence, it extracts all eligible events, and do temperal ordering of them;

    :param TE_ids: list [ list [s_id,s_id2,e_id,e_id2],  [s_id',s_id2',e_id',e_id2'], ...    ]
    :param sentences: list [sentence, sentence2, .. ]
    :param srl: alllennlp srl output, read into dataframe
    :param event_coref: the event_coref file defined in our pipeline, see slide

    :return: the data format required by ECONET TE_ordering pipeline
    """

    d = dict()
    for i in (range(len(TE_ids))):
        s_id,e_id,s_id2,e_id2 = TE_ids[i]
        v = create_data_instance(s_id,s_id2,e_id,e_id2, sentences, srl, event_coref)
        d['L_' + str(i)] = v
    return d

def get_auxis(coref_args_df):
    """
    Outputs new list indicating whether its corresonding row in df is an auxiliary verb
    :param coref_args_df:
    :return: list indicating whether its corresonding row in df is an auxiliary verb
    """

    aux_ls = ['is', 'are','was','were','be','been','has','have','had','going','can','could','will','would','shall','should',
              'may','might','must','do','does','did']
    aux = []
    for i in range(coref_args_df.shape[0]):
        line = coref_args_df.iloc[i]
        verb = line['verb']
        if verb in aux_ls:
            aux.append(True)
        else:
            aux.append(False)
    return aux

# modify this file to come up with those ids directly from srl outputs;



def create_TE_ids_quotation(srl, sentences):
    """
    Create the TE_ids data for the main data creation function;
    modify to include quotation and main event information for each verb. 
    """
    sent_ids, verb_ids, verbs = [],[],[]
    for sent_id in range(len(srl)):
        srl_ls = srl[sent_id]['verbs']
        for verb_id, srl_dict in enumerate(srl_ls):
            sent_ids.append(sent_id)
            verb_ids.append(verb_id)
            verbs.append(srl_dict['verb'])
    # create this equivalent dataframe, for psuedo coref_args_df;
    coref_args_df = pd.DataFrame(np.array([sent_ids, verb_ids, verbs]).transpose(), columns=['sentence_id','verb_id','verb'] )
    auxs = get_auxis(coref_args_df)
    coref_args_df['auxi'] = auxs
    #     coref_sent_ids = list(set(coref_args_df['sentence_id']))
    d = dict()
    # get rid of all auxilliary verbs;
    coref_args_df = coref_args_df[coref_args_df['auxi']==False]
    for i in range(coref_args_df.shape[0]):
        row = coref_args_df.iloc[i]
        s_id = int(row['sentence_id'])
        if s_id not in d:
            d[s_id] = set()
        d[s_id].add(int(row['verb_id']))
    prev = None

    TE_ids = []
    # print("the length of dictionary", len(d.keys()))
    for s_id in d.keys():
        verb_ids = list(d[s_id])
        for v_id in verb_ids:
            if prev == None:
                prev = [s_id, v_id]
                continue
            TE_ids.append( [prev[0], prev[1], s_id, v_id] )
            prev = [s_id, v_id]
    return TE_ids


def check_quotation(words, v_id):
    """
    Check whether specific verb is in a quotation

    :param words:
    :param v_id:
    :return:
    """
    befores = words[:v_id]
    cnt = befores.count('"')
    if cnt % 2 == 0:
        return False
    return True

def create_TE_ids_from_coref(event_coref):
    """
    Create the TE_ids data for the main data creation function;
    This is really important, bc, it decides which event eventually goes to the TE-ordering algorithm
    :param event_coref: event_coref df
    :return:
            [
            [sentence_id1, event_id1, sentence_id2, event_id2]
            ....
            ]
    """
    event_coref = event_coref[event_coref['event_label']==1]
    event_coref = event_coref.loc[ (event_coref['subj_coref_ids'].isnull()==False) | (event_coref['dobj_coref_ids'].isnull()==False)]

    d = dict()
    for i in range(event_coref.shape[0]):
        row = event_coref.iloc[i]
        s_id = int(row['sentence_id'])
        if s_id not in d:
            d[s_id] = set()
        d[s_id].add(int(row['verb_id']))
    prev = None

    TE_ids = []
    print("the length of dictionary", len(d.keys()))
    for s_id in d.keys():
        verb_ids = list(d[s_id])
        for v_id in verb_ids:
            if prev == None:
                prev = [s_id, v_id]
                continue
            TE_ids.append( [prev[0], prev[1], s_id, v_id] )
            prev = [s_id, v_id]
    return TE_ids




def create_TE_ids(srl, event_coref, no_quote=False):
    """
    Create the TE_ids data for the main data creation function;
    This is really important, bc, it decides which event eventually goes to the TE-ordering algorithm

    :param srl: the srl df
    :param event_coref: event_coref df
    :param no_quote: whether run event_TE ordering for event in a quotation marks
    :return:
            [
            [sentence_id1, event_id1, sentence_id2, event_id2]
            ....
            ]
    """

    sent_ids, verb_ids, verbs, quotation = [],[],[],[]
    for sent_id in range(len(srl)):
        srl_ls = srl[sent_id]['verbs']
        for verb_id, srl_dict in enumerate(srl_ls):
            #we probly only want to take events that are verbs;
            if 'B-V' not in srl_dict['tags']:
                continue

            sent_ids.append(sent_id)
            verb_ids.append(verb_id)
            verbs.append(srl_dict['verb'])
            v_loc = srl_dict['tags'].index('B-V')
            quotation.append(check_quotation(srl[sent_id]['words'], v_loc ))
    #create this equivalent dataframe, for psuedo coref_args_df;
    coref_args_df = pd.DataFrame(np.array([sent_ids, verb_ids, verbs, quotation]).transpose(), columns=['sentence_id','verb_id','verb','in_quote'] )
    auxs = get_auxis(coref_args_df)
    coref_args_df['auxi'] = auxs
    #     coref_sent_ids = list(set(coref_args_df['sentence_id']))
    d = dict()
    # Do not run TE_extraction on auxilliary verbs or words in quotation marks;
    if no_quote:
        coref_args_df = coref_args_df[coref_args_df['in_quote'] == 'False']
    coref_args_df = coref_args_df[coref_args_df['auxi']==False]
    for i in range(coref_args_df.shape[0]):
        row = coref_args_df.iloc[i]
        s_id = int(row['sentence_id'])
        if s_id not in d:
            d[s_id] = set()
        d[s_id].add(int(row['verb_id']))
    prev = None

    TE_ids = []
    print("the length of dictionary", len(d.keys()))
    for s_id in d.keys():
        verb_ids = list(d[s_id])
        for v_id in verb_ids:
            if prev == None:
                prev = [s_id, v_id]
                continue
            TE_ids.append( [prev[0], prev[1], s_id, v_id] )
            prev = [s_id, v_id]
    return TE_ids


def get_core_data(dir='../may18_data/', story='cinderella-or-the-little-glass-slipper'):
    """
    Get the core data for a specific story, including srl results, event_coref file, and sentences file.
    :param dir:
    :param story:
    :return: the three dataframe loading the three files
    """
    story_dir = dir + story + '/'
    if path.isfile(path.join(story_dir, story +'.srl.json')):
        srl_path = path.join(story_dir, story +'.srl.json')
    else:
        srl_path = path.join(story_dir, story + '.srl.json')
    sentences_path = path.join(story_dir, story + '.sentences.csv')
    characters_path = path.join(story_dir, story + '.actions_args.csv')

    with open(srl_path, 'r') as json_file:
        srl = json.load(json_file)
    sentences_df = pd.read_csv(sentences_path)
    sentences = sentences_df['text'].tolist()
    #optionally output event_coref if the path exsits
    if path.exists(characters_path):
        event_coref = pd.read_csv(characters_path)
    else:
        event_coref = None
    return srl, sentences, event_coref



def get_allen_srl(dir,story):
    srls = []
    # predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.11.19.tar.gz")
    predictor = pretrained.load_predictor("structured-prediction-srl")
    #get the sentences to run srl with
    sentences_path = path.join(dir, story + '.sentences.csv')
    sentences_df = pd.read_csv(sentences_path)
    sentences = sentences_df['text'].tolist()

    for sent in sentences:
        srl1 = predictor.predict(
            sentence=sent
        )
        srls.append(srl1)
    srl_dir = dir
    with open(path.join(srl_dir,story+'.srl.json'), "w") as final:
        json.dump(srls, final)

def save_output_df(story_dir,srl,sentences,event_coref,major_event=False):
    """
    Saves the final output_df to the respective story folder.
    :param dir:
    :param story:
    :param srl:
    :param sentences:
    :param major: whether it is a major event df
    :return:
    """
    log = create_TE_ids(srl, sentences)
    log = create_TE_ids_from_coref(event_coref)
    rows = []
    for tup in log:
        s_id,e_id,s_id2, e_id2 = tup
        row = [sentences[s_id], srl[s_id]['verbs'][e_id]['verb'], None, sentences[s_id2], srl[s_id2]['verbs'][e_id2]['verb'], s_id,s_id2,e_id,e_id2]
        rows.append(row)
    output = pd.DataFrame(rows, columns=['sentence1', 'event1','te_rel', 'sentence2', 'event2', 's_id','s_id2','e_id','e_id2' ])
    
    if major_event:
        file_name = "major_TE_output.csv"
    else:
        file_name = "TE_output.csv"
    
    output.to_csv(path.join(story_dir, file_name), index=None)


from src.major_actions_args import get_major_actions
def get_data(dir='../new_test/', story='cinderella-or-the-little-glass-slipper', get_srl=False, major_event=False):
    """
    :param dir: data dir
    :param story: story-name
    :param get_srl:
                    True if you want to create new srl files given the sentences file;
                    False if the srl file are already available and in-place.
    :return: TE_data instance to run TE inference
    """
    if get_srl:
        get_allen_srl(dir,story)
    #the third is character_df to be filled up and utilized later
    srl, sentences, event_coref = get_core_data(dir,story)

    # TE_ids = create_TE_ids(srl, event_coref)
    TE_ids = create_TE_ids_from_coref(event_coref)
    
    data = create_data_instances_multi_events(TE_ids,sentences,srl,event_coref)

    #create the story_dir
    story_dir = path.join(dir,story)
    if not path.exists(story_dir):
        os.makedirs(story_dir)
    #save the data.pickle file
    with open(path.join(story_dir, "data.pickle"), 'wb') as f:
        pickle.dump(data, f)
    #save the output file;
    save_output_df(story_dir,srl,sentences,event_coref)
    
    if major_event:
        story_text = ' '.join(sentences) 
        # major_actions_df = get_major_actions(event_coref,story_text)
        major_actions_df = pd.read_csv(path.join(story_dir, "major_actions_args.csv"))
        major_TE_ids = create_TE_ids_from_coref(major_actions_df)
        major_data = create_data_instances_multi_events(major_TE_ids,sentences,srl,major_actions_df)
        with open(path.join(story_dir, "major_data.pickle"), 'wb') as f:
            pickle.dump(major_data, f)
        # major_actions_df.to_csv(path.join(story_dir, "major_actions_args.csv"),index=None)

        save_output_df(story_dir,srl,sentences,major_actions_df, major_event=True)
    
    return data



if __name__ == "__main__":
    stories = ['ali-baba-and-forty-thieves', 'old-dschang','cinderella-or-the-little-glass-slipper'
        ,'bamboo-cutter-moon-child','leelinau-the-lost-daughter','the-dragon-princess']
    # stories= ['the-dragon-princess']
    for i in range(6):
        get_data('../may18_data/', stories[i], get_srl=False)

