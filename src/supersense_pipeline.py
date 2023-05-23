# Standard Library
import json
import os

# Third Party
import numpy as np
import pandas as pd

# Local
from src.utils import read_story_txt_file

class SupersensePipeline:
    def __init__(self, story_id: str, story_text: str, 
                 tokens_df: pd.DataFrame, supersense_df: pd.DataFrame, actions_df: pd.DataFrame, 
                 pipeline_dir: str,
                 keep_all_verbs = False):
        self.story_id = story_id
        self.pipeline_dir = pipeline_dir 

        self.supersense_df = supersense_df 
        self.actions_df = actions_df
        self.tokens_df = tokens_df
        self.story_text = story_text

        if 'supersense' not in self.actions_df.columns or 'event_label' not in self.actions_df.columns:
            self.actions_df = self.get_merged_supersense_actions_df(keep_all_verbs)
            self.save_supersense_to_actions_args_csv()

    @classmethod
    def from_files(cls, story_id: str, pipeline_dir: str, keep_all_verbs = False):
        story_dir_with_prefix = os.path.join(pipeline_dir,story_id, story_id)
        tokens_file = story_dir_with_prefix + '.tokens'
        supersense_file = story_dir_with_prefix + '.supersense'
        actions_file = story_dir_with_prefix + '.actions_args.csv'
        story_txt_file = story_dir_with_prefix + '.txt'

        tokens_df = pd.read_csv(tokens_file, sep='\t')
        supersense_df = pd.read_csv(supersense_file, sep='\t')
        actions_df = pd.read_csv(actions_file)
        story_text = read_story_txt_file(story_txt_file)

        return cls(story_id, story_text, tokens_df, supersense_df, actions_df, pipeline_dir, keep_all_verbs)

    def get_merged_supersense_actions_df(self, keep_all_verbs):
        self.supersense_df['event_label'] = self.get_action_verb_supersense_labels(keep_all_verbs)
        self.supersense_df = self.supersense_df[self.supersense_df['event_label'] == 1].copy()

        self.supersense_df.loc[:, 'start_byte'] = self.supersense_df.apply(self.get_start_bytes, tokens_df = self.tokens_df, axis = 1)
        self.supersense_df.loc[:, 'end_byte'] = self.supersense_df.apply(self.get_end_bytes, tokens_df = self.tokens_df, axis = 1)

        supersense_actions_df = self.actions_df.merge(self.supersense_df, left_on = ['verb_start_byte_text', 'verb_end_byte_text'], right_on = ['start_byte', 'end_byte'], how = 'inner')
        supersense_actions_df.drop(columns = ['start_token', 'end_token', 'start_byte', 'end_byte', 'text'], inplace = True)

        return supersense_actions_df

    def get_action_verb_supersense_labels(self, keep_all_verbs):
        if keep_all_verbs == False:
            return np.where((self.supersense_df['supersense_category'].str[:4] == 'verb') & (self.supersense_df['supersense_category'] != 'verb.stative'), 1, 0)
        elif keep_all_verbs == True:
            return np.where((self.supersense_df['supersense_category'].str[:4] == 'verb'), 1, 0)

    def set_start_end_bytes_of_events(self):
        self.supersense_df['start_byte'] = None
        self.supersense_df['end_byte'] = None

        for i, row in self.supersense_df.iterrows():
            token_range = list(range(row['start_token'], row['end_token'] + 1))
            char_tokens_df = self.tokens_df[self.tokens_df['token_ID_within_document'].isin(token_range)]
            self.supersense_df.iloc[i, 5] = int(char_tokens_df['byte_onset'].tolist()[0])
            self.supersense_df.iloc[i, 6] = int(char_tokens_df['byte_offset'].tolist()[-1])
            # if (self.story_text[start_byte:end_byte] != row['text']):
            #     self.supersense_df.iloc[i, 'start_byte'] = start_byte
            #     self.supersense_df.iloc[i, 'end_byte'] = end_byte
            #     start_end_bytes_of_events.append([start_byte, end_byte, self.story_text[start_byte:end_byte]]) 
            # else:
            #     start_end_bytes_of_events.append([start_byte, end_byte, row['text']])
            
    @staticmethod
    def get_start_bytes(row, tokens_df):
        token_range = list(range(row['start_token'], row['end_token'] + 1))
        char_tokens_df = tokens_df[tokens_df['token_ID_within_document'].isin(token_range)]
        return int(char_tokens_df['byte_onset'].tolist()[0])

    @staticmethod
    def get_end_bytes(row, tokens_df):
        token_range = list(range(row['start_token'], row['end_token'] + 1))
        char_tokens_df = tokens_df[tokens_df['token_ID_within_document'].isin(token_range)]
        return int(char_tokens_df['byte_offset'].tolist()[-1])

    def save_supersense_to_actions_args_csv(self):
        story_dir_with_prefix = self.pipeline_dir + self.story_id + '/' + self.story_id
        actions_file = story_dir_with_prefix + '.actions_args.csv'
        self.actions_df.to_csv(actions_file, index = False)
        print('Saving {} action_args CSV to: {}'.format(self.story_id, actions_file))