import json
import re

import pandas as pd


#SRL results

def check_B_arg(text: str):
    return re.search(r'B-ARG\d+', text) is not None
def check_I_arg(text: str):
    return re.search(r'I-ARG\d+', text) is not None
def check_arg(text: str):
    return re.search(r'ARG\d+', text) is not None

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

def get_arg_attributes(verb: dict, verb_id: int, words: list, tokens_start_end_bytes: dict, sentence_id: int, sentences: list):
    args = []
    sentence = sentences[sentence_id]
    for i, tag in enumerate(verb['tags']):
        tokens = []
        arg_token_start_end_bytes = []
        if check_B_arg(tag):
            arg_token_start_end_bytes.append((tokens_start_end_bytes[i]['start_byte'], tokens_start_end_bytes[i]['end_byte']))
            tokens.append(words[i])
            j = i + 1
            while (j < len(verb['tags'])) and check_I_arg(verb['tags'][j]):
                tokens.append(words[j])
                arg_token_start_end_bytes.append((tokens_start_end_bytes[j]['start_byte'], tokens_start_end_bytes[j]['end_byte']))
                j += 1
            # Get start and end bytes of tokens
            start_byte = tokens_start_end_bytes[i]['start_byte']
            end_byte = tokens_start_end_bytes[j-1]['end_byte']

            arg_dict = {}
            arg_dict['sentence_id'] = sentence_id
            arg_dict['verb_id'] = verb_id
            arg_dict['verb'] = verb['verb']
            arg_dict['tag'] = tag
            arg_dict['text'] = sentence[start_byte:end_byte]
            arg_dict['start_byte'] = start_byte
            arg_dict['end_byte'] = end_byte
            # arg_dict['tokens'] = tokens
            # arg_dict['tokens_start_end_bytes'] = arg_token_start_end_bytes
            args.append(arg_dict)
    return args

def has_noun_argument(argument, pos_tags_sentence):
    skip_i = 0
    for i, token_start_end_bytes in enumerate(argument['tokens_start_end_bytes']):
        if i < len(argument['tokens_start_end_bytes']) - 1:
            if '-' in argument['tokens'][i+1]:
                hyphen_token_start_end_bytes = (token_start_end_bytes[0], argument['tokens_start_end_bytes'][i+2][1])
                tag = pos_tags_sentence[(hyphen_token_start_end_bytes)][1]
                if tag == 'NOUN' or tag == 'PRON':
                    return True
                skip_i = 2
            elif skip_i > 0:
                skip_i -= 1
            else:
                tag = pos_tags_sentence[(token_start_end_bytes)][1]
                if tag == 'NOUN' or tag == 'PRON':
                    return True
    return False
def coref_in_arguments(coref_ids: str):
    all_coref_ids = ''.join(coref_ids)
    if all_coref_ids == '':
        return False
    else:
        return True


def get_core_data(dir='../new_test/', story='cinderella-or-the-little-glass-slipper'):
    #get files first
    # srl_path = './new_test/srl/cinderella-or-the-little-glass-slipper.srl.json'
    # sentences_path = './new_test/sentence/cinderella-or-the-little-glass-slipper.sentences.csv'
    # characters_path = './new_test/sentences_characters/cinderella-or-the-little-glass-slipper.sentences_characters.csv'
    srl_path = dir + 'srl/' + story + '.srl.json'
    sentences_path = dir + 'sentence/' + story + '.sentences.csv'
    characters_path = dir + 'sentences_characters/' + story + '.sentences_characters.csv'

    with open(srl_path, 'r') as json_file:
        srl = json.load(json_file)
    sentences_df = pd.read_csv(sentences_path)
    sentences = sentences_df['text'].tolist()
    characters_df = pd.read_csv(characters_path)
    return srl, sentences, characters_df





def get_coref_df(dir='../new_test/', story='cinderella-or-the-little-glass-slipper'):
    srl, sentences, characters_df = get_core_data(dir,story)

    all_sentences_args = []
    for i, sentence in enumerate(sentences):
        for j, srl_verb in enumerate(srl[i]['verbs']):
            verb = srl_verb['verb']
            token_start_end_bytes = get_tokens_start_end_bytes(srl[i]['words'], sentence)
            attributes = get_arg_attributes(srl_verb, j, srl[i]['words'], token_start_end_bytes, i, sentences)
            all_sentences_args += attributes

    args_df = pd.DataFrame(all_sentences_args)
    args_df['coref_ids'] = ''
    for i, row in args_df.iterrows():
        sentence_id = row['sentence_id']
        arg_start = row['start_byte']
        arg_end = row['end_byte']
        sen_char_df = characters_df[characters_df['sentence_id'] == sentence_id]
        args_char_df = sen_char_df[(sen_char_df['start_byte_in_sentence'] >= arg_start) & (sen_char_df['end_byte_in_sentence'] <= arg_end)]
        if len(args_char_df) > 0:
            args_df.loc[i, 'coref_ids'] = ','.join((args_char_df['coref_id'].astype('str').tolist()))
    args_df = args_df.merge(args_df.groupby(['sentence_id', 'verb_id'])['coref_ids'].apply(coref_in_arguments).reset_index().rename(columns = {'coref_ids': 'coref_in_args'}), on = ['sentence_id', 'verb_id'], how = 'left')
    return args_df