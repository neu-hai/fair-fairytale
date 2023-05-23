# Standard Library
import json
import re

# Third Party
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.tokenize import TreebankWordTokenizer as twt
import pandas as pd
# Local
from src.utils import read_story_txt_file

class ActionArgsPipeline:
    def __init__(self, story_id: str, story_text: str, sentences_df: pd.DataFrame, characters_df: pd.DataFrame, srl_json: dict, pipeline_dir: str):
        self.story_id = story_id
        self.pipeline_dir = pipeline_dir 

        self.story_text = story_text
        self.sentences_df = sentences_df
        self.characters_df = characters_df
        self.srl = srl_json
        self.sentences = self.sentences_df['text'].tolist()
        self.get_additional_argument = True

        # Assert that SRL and sentence lists are of equal size.
        if len(self.srl) == len(self.sentences):
            self.valid = True
            self.run_pipeline()
        else:
            print(len(self.srl))
            print(len(self.sentences))
            self.valid = False
            print("SRL and sentences files are not of the same length. Cannot extract actions and arguments.")

    @classmethod
    def from_files(cls, story_id: str, pipeline_dir: str):
        story_dir_with_prefix = pipeline_dir + story_id + '/' + story_id
        sentences_path = story_dir_with_prefix + '.sentences.csv'
        characters_path = story_dir_with_prefix + '.sentences_characters.csv'
        story_path = story_dir_with_prefix + '.txt'
        srl_path = story_dir_with_prefix + '.srl.json'
        story_text = read_story_txt_file(story_path)
        sentences_df = pd.read_csv(sentences_path)
        characters_df = cls.get_characters_df(characters_path)
        srl = cls.get_srl(srl_path)
        return cls(story_id, story_text, sentences_df, characters_df, srl, pipeline_dir)

    @classmethod
    def get_characters_df(self, characters_path: str):
        characters_df = pd.read_csv(characters_path)
        characters_df = characters_df[characters_df['overlap'] == 0]
        return characters_df

    @classmethod
    def get_srl(self, srl_path: str):
        with open(srl_path, 'r') as json_file:
            srl = json.load(json_file)
        return srl

    def get_pos_tags(self):
        pos_tags_sentences = []

        for j, sentence in enumerate(self.sentences):
            pos_tags_sentence = {}
            pos_tags = pos_tag(word_tokenize(sentence), tagset='universal')
            token_spans = list(twt().span_tokenize(sentence))
            if len(pos_tags) == len(token_spans):
                for i, span in enumerate(token_spans):
                    pos_tags_sentence[(span[0], span[1])] = (pos_tags[i][0], pos_tags[i][1])
            else:
                for i, span in enumerate(token_spans):
                    if sentence[span[0]:span[1]] ==  pos_tags[i][0]:
                        pos_tags_sentence[(span[0], span[1])] = (pos_tags[i][0], pos_tags[i][1])
                    else:
                        pos_tags_sentence[(span[0], span[1])] = (pos_tags[i+1][0], pos_tags[i+1][1])
            pos_tags_sentences.append(pos_tags_sentence)

        return pos_tags_sentences

    def get_tokens_start_end_bytes(self, tokens, sentence):
        token_start_end_bytes_sentence = []
        
        s = 0
        for i, token in enumerate(tokens):
            token_start_end_bytes = {}
            
            token_start_end_bytes['token'] = token
            token_start_end_bytes['i'] = i
            
            start_byte = s
            end_byte = s + len(token)
            
            if sentence[start_byte] == ' ':
                start_byte += 1
                end_byte += 1
            
            #print(start_byte, end_byte)
            #print(sentence[start_byte])
            #print(sentence[end_byte])
            #print("mapping :", token, sentence[start_byte:end_byte])
            #print("\n")
            
            if token != sentence[start_byte:end_byte]:
                print('Incorrect mapping of token to sentence bytes.')
                print(i, token, sentence[start_byte:end_byte])
            
            token_start_end_bytes['start_byte'] = start_byte
            token_start_end_bytes['end_byte'] = end_byte
        
            token_start_end_bytes_sentence.append(token_start_end_bytes)
            
            s = end_byte
        
        return token_start_end_bytes_sentence

    def get_arg_attributes(self, verb: dict, verb_id: int, words: list, tokens_start_end_bytes: dict, sentence_id: int):
        args = []
        sentence = self.sentences[sentence_id]
        verb_start_byte = None
        verb_end_byte = None
        for i, tag in enumerate(verb['tags']):
            if check_verb(tag):
                verb_start_byte = int(tokens_start_end_bytes[i]['start_byte'])
                verb_end_byte = int(tokens_start_end_bytes[i]['end_byte'])
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
                arg_dict['verb_start_byte'] = verb_start_byte
                arg_dict['verb_end_byte'] = verb_end_byte
                arg_dict['tag'] = tag
                arg_dict['text'] = sentence[start_byte:end_byte]
                arg_dict['arg_start_byte'] = start_byte
                arg_dict['arg_end_byte'] = end_byte
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

    def get_all_sentences_args(self):
        all_sentences_args = []
        for i, sentence in enumerate(self.sentences):
            for j, srl_verb in enumerate(self.srl[i]['verbs']):
                verb = srl_verb['verb']
                token_start_end_bytes = self.get_tokens_start_end_bytes(self.srl[i]['words'], sentence)
                attributes = self.get_arg_attributes(srl_verb, j, self.srl[i]['words'], token_start_end_bytes, i)
                all_sentences_args += attributes
        return all_sentences_args 

    def get_coref_ids_in_actions_df(self, actions_df):
        actions_df['coref_ids'] = ''
        for i, row in actions_df.iterrows():
            sentence_id = row['sentence_id']
            arg_start = row['arg_start_byte']
            arg_end = row['arg_end_byte']
            sen_char_df = self.characters_df[self.characters_df['sentence_id'] == sentence_id]
            args_char_df = sen_char_df[(sen_char_df['start_byte_in_sentence'] >= arg_start) & (sen_char_df['end_byte_in_sentence'] <= arg_end)]
            if len(args_char_df) > 0:
                actions_df.loc[i, 'coref_ids'] = ','.join((args_char_df['coref_id'].astype('str').tolist()))
        actions_df = actions_df.merge(actions_df.groupby(['sentence_id', 'verb_id'])['coref_ids'].apply(self.coref_in_arguments).reset_index().rename(columns = {'coref_ids': 'coref_in_args'}), on = ['sentence_id', 'verb_id'], how = 'left')
        return actions_df
    
    def subset_verbs_with_corefs_in_actions_df(self, actions_df):
        return actions_df[actions_df['coref_in_args'] == True]

    def set_arg_type_in_actions_df(self):
        self.actions_df.loc[:, 'arg_id'] = pd.to_numeric(self.actions_df['tag'].str[-1])
        self.actions_df = self.actions_df.merge(self.actions_df.groupby(['sentence_id', 'verb_id']).arg_id.count().reset_index().rename(columns = {'arg_id': 'args_n'}), on = ['sentence_id', 'verb_id'], how = 'left')
        self.actions_df = self.actions_df.merge(self.actions_df.groupby(['sentence_id', 'verb_id']).arg_id.min().reset_index().rename(columns = {'arg_id': 'args_min'}), on = ['sentence_id', 'verb_id'], how = 'left')

        for i, row in self.actions_df.iterrows():
            if row['arg_start_byte'] < row['verb_start_byte']:
                self.actions_df.loc[i,'arg_type'] = 'subject'
            elif row['arg_start_byte'] > row['verb_start_byte']:
                self.actions_df.loc[i,'arg_type'] = 'direct_object'
            else:
                # print("Neither subject nor direct object:")
                # print(row['sentence_id'], row['verb_id'], row['verb'])
                # print("arg_start_byte: {}, verb_start_byte: {}".format(row['arg_start_byte'], row['verb_start_byte']))
                # print("arg_end_byte: {}, verb_end_byte: {}".format(row['arg_end_byte'], row['verb_end_byte']))
                self.actions_df.loc[i,'arg_type'] = 'error'
        self.actions_df = self.actions_df[self.actions_df['arg_type'] != 'error']


    def convert_long_to_wide_actions_df(self):
        arg_type_df = self.actions_df.groupby(['sentence_id', 'verb_id', 'arg_type'])['coref_ids'].apply(lambda x: ','.join(x)).reset_index().drop_duplicates()
        arg_type_df = self.actions_df[['sentence_id', 'verb_id', 'verb', 'arg_start_byte', 'arg_end_byte', 'arg_type']].merge(arg_type_df, on = ['sentence_id', 'verb_id', 'arg_type'], how = 'left').drop_duplicates()
        arg_type_df.drop_duplicates(['sentence_id', 'verb_id', 'arg_type'], inplace = True)
        
        # print("arg_type_df")
        # print(arg_type_df.head())
        # print(arg_type_df.columns)

        pivot_df = arg_type_df[['sentence_id', 'verb_id', 'arg_start_byte', 'arg_end_byte', 'arg_type', 'coref_ids']].drop_duplicates()
        # print("pivot_df")
        # print(pivot_df.head())
        # print(pivot_df.columns)
        # print(pivot_df.dtypes)
        # print(pivot_df['arg_type'].value_counts(dropna = False))

        final_df = pivot_df.pivot(index = ['sentence_id', 'verb_id'], columns = 'arg_type', values = ['arg_start_byte', 'arg_end_byte', 'coref_ids'])
        # print("final_df")
        # print(final_df.head())

        # print(final_df.columns)
        final_df.columns = final_df.columns.to_flat_index()
        # print(final_df.columns)

        final_df.rename(columns = {('arg_start_byte', 'direct_object'): 'dobj_start_byte',
                                         ('arg_start_byte', 'subject'): 'subj_start_byte',
                                         ('arg_end_byte', 'direct_object'): 'dobj_end_byte',
                                         ('arg_end_byte', 'subject'): 'subj_end_byte',
                                         ('coref_ids', 'direct_object'): 'dobj_coref_ids',
                                         ('coref_ids', 'subject'): 'subj_coref_ids'}, inplace = True)
        final_df.reset_index(inplace = True)
        self.actions_df = self.actions_df[['sentence_id', 'verb_id', 'verb', 'verb_start_byte', 'verb_end_byte']].merge(final_df, on = ['sentence_id', 'verb_id'], how = 'left').drop_duplicates()

    def clean_actions_df(self):
        self.actions_df.fillna('', inplace = True)
        self.actions_df['subj_coref_ids'] = self.actions_df['subj_coref_ids'].apply(self.clean_coref_ids)
        self.actions_df['dobj_coref_ids'] = self.actions_df['dobj_coref_ids'].apply(self.clean_coref_ids)

    def set_text_offsets_actions_df(self):
        ordered_columns = ['sentence_id', 'verb_id', 'verb', 'subj_coref_ids', 'subj_start_byte', 'subj_end_byte', 'dobj_coref_ids', 'dobj_start_byte', 'dobj_end_byte']
        final_text_offsets_df = self.actions_df.merge(self.sentences_df[['sentence_id', 'start', 'end']], on = 'sentence_id', how = 'left')
        bytes_to_numeric = ['verb_start_byte', 'verb_end_byte', 'dobj_start_byte', 'subj_start_byte', 'dobj_end_byte', 'subj_end_byte']
        final_text_offsets_df[bytes_to_numeric] = final_text_offsets_df[bytes_to_numeric].apply(pd.to_numeric, errors='coerce')

        # print(final_text_offsets_df.head())
        # print(final_text_offsets_df.columns)
        # print(final_text_offsets_df.dtypes)
        # print(final_text_offsets_df[bytes_to_numeric].head())

        final_text_offsets_df['verb_start_byte_text'] = final_text_offsets_df['verb_start_byte'] + final_text_offsets_df['start']
        final_text_offsets_df['verb_end_byte_text'] = final_text_offsets_df['verb_end_byte'] + final_text_offsets_df['start']
        final_text_offsets_df['subj_start_byte_text'] = final_text_offsets_df['subj_start_byte'] + final_text_offsets_df['start']
        final_text_offsets_df['subj_end_byte_text'] = final_text_offsets_df['subj_end_byte'] + final_text_offsets_df['start']
        final_text_offsets_df['dobj_start_byte_text'] = final_text_offsets_df['dobj_start_byte'] + final_text_offsets_df['start']
        final_text_offsets_df['dobj_end_byte_text'] = final_text_offsets_df['dobj_end_byte'] + final_text_offsets_df['start']

        self.actions_df = final_text_offsets_df[['sentence_id', 'verb_id', 'verb', 'verb_start_byte', 'verb_end_byte', 'verb_start_byte_text', 'verb_end_byte_text',
                               'subj_coref_ids', 'subj_start_byte', 'subj_end_byte', 'subj_start_byte_text', 'subj_end_byte_text',
                               'dobj_coref_ids', 'dobj_start_byte', 'dobj_end_byte', 'dobj_start_byte_text', 'dobj_end_byte_text']]


    def save_actions_df_to_csv(self):
        action_args_file = self.pipeline_dir + self.story_id + '/' + self.story_id + '.actions_args.csv'
        self.actions_df.to_csv(action_args_file, index = False)
        print('Saving {} action_args CSV to: {}'.format(self.story_id, action_args_file))

    @staticmethod
    def coref_in_arguments(coref_ids: str):
        all_coref_ids = ''.join(coref_ids)
        if all_coref_ids == '':
            return False
        else:
            return True

    @staticmethod
    def clean_coref_ids(text: str):
        if len(text) > 0:
            if text == ',':
                return ''
            if text[0] == ',':
                text = text[1:]
            if text[-1] == ',':
                text = text[:-1]
            text = ','.join(list(set(text.split(','))))
            return text


    def run_pipeline(self):
        all_sentences_args = self.get_all_sentences_args()
        assert len(self.srl) == len(self.sentences)

        actions_df = self.get_coref_ids_in_actions_df(pd.DataFrame(all_sentences_args))

        self.actions_df = self.subset_verbs_with_corefs_in_actions_df(actions_df)

        self.set_arg_type_in_actions_df()

        self.convert_long_to_wide_actions_df()

        self.clean_actions_df()

        self.set_text_offsets_actions_df()

        if self.get_additional_argument:
            self.actions_df = get_additional_arguments(self.actions_df, self.srl, self.sentences)

        self.save_actions_df_to_csv()

# SRL Argument Functions
def check_B_arg(text: str):
    return re.search(r'B-ARG\d+', text) is not None

def check_I_arg(text: str):
    return re.search(r'I-ARG\d+', text) is not None

def check_arg(text: str):
    return re.search(r'ARG\d+', text) is not None

def check_verb(text: str):
    return re.search(r'\w-V', text) is not None

def check_M_arg(text: str):
    return re.search(r'\D-ARGM\D', text)

def get_tokens_start_end_bytes(tokens, sentence):
    token_start_end_bytes_sentence = []

    s = 0
    for i, token in enumerate(tokens):
        token_start_end_bytes = {}

        token_start_end_bytes['token'] = token
        token_start_end_bytes['i'] = i

        start_byte = s
        end_byte = s + len(token)

        if sentence[start_byte] == ' ':
            start_byte += 1
            end_byte += 1

        #print(start_byte, end_byte)
        #print(sentence[start_byte])
        #print(sentence[end_byte])
        #print("mapping :", token, sentence[start_byte:end_byte])
        #print("\n")

        if token != sentence[start_byte:end_byte]:
            print('Incorrect mapping of token to sentence bytes.')
            print(i, token, sentence[start_byte:end_byte])

        token_start_end_bytes['start_byte'] = start_byte
        token_start_end_bytes['end_byte'] = end_byte

        token_start_end_bytes_sentence.append(token_start_end_bytes)

        s = end_byte

    return token_start_end_bytes_sentence

def get_additional_arguments(df, srl, sentences):
    adobj_start_bytes = []
    adobj_end_bytes = []
    offsets = []
    for idx in range(df.shape[0]):
        row = df.iloc[idx]
        s_id, v_id = row['sentence_id'], row['verb_id']
        token_start_end_bytes = get_tokens_start_end_bytes(srl[s_id]['words'], sentences[s_id])
        sentence, verb = sentences[s_id], srl[s_id]['verbs'][v_id]
        dobj_end = row['dobj_end_byte']
        tags = verb['tags']
        adobj_start_byte, adobj_end_byte = None, None

        start_fixed = False
        for i, tag in enumerate(tags):
            # if it's a modifier argument after the verb, we add modifier argument columns
            if check_M_arg(tag) and token_start_end_bytes[i]['start_byte'] > dobj_end:
                if not start_fixed:
                    adobj_start_byte, adobj_end_byte = token_start_end_bytes[i]['start_byte'], token_start_end_bytes[i][
                        'end_byte']
                    start_fixed = True
                else:
                    adobj_end_byte = token_start_end_bytes[i]['end_byte']

                j = i + 1
                while j < len(tags) and check_M_arg(tags[j]):
                    adobj_end_byte = token_start_end_bytes[j]['end_byte']
                    j += 1

        adobj_start_bytes.append(adobj_start_byte)
        adobj_end_bytes.append(adobj_end_byte)
        if adobj_start_byte:
            offsets.append(row['subj_start_byte_text'] - row['subj_start_byte'])
        else:
            offsets.append(None)

    df['adobj_start_bytes'] = adobj_start_bytes
    df['adobj_end_bytes'] = adobj_end_bytes
    df['adobj_start_bytes_text'] = [adobj_start_bytes[i] + offsets[i] if offsets[i] else None for i in
                                    range(len(offsets))]
    df['adobj_end_bytes_text'] = [adobj_end_bytes[i] + offsets[i] if offsets[i] else None for i in range(len(offsets))]
    return df
