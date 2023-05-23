import os
import re
import csv

from typing import List

import numpy as np
import pandas as pd

from src.sentences import split_sentences 
from src.utils import read_story_txt_file

class TemporalEventsCharactersPipeline:
    def __init__(self, story_id: str, characters_df: pd.DataFrame, actions_args_df: pd.DataFrame, ranks_df: pd.DataFrame, pipeline_dir: str, 
                 major_actions_args_df = None, major_ranks_df = None):
        self.story_id = story_id
        self.pipeline_dir = pipeline_dir

        self.characters_df = self.get_clean_characters_df(characters_df)
        self.actions_args_df = actions_args_df.rename(columns = {'verb_id': 'event_id'})
        self.ranks_df = ranks_df

        # print(self.ranks_df)

        # print(self.actions_args_df.head())

        self.add_temporal_ranks_to_actions_args_df(major_event = False)
        # print(self.actions_args_df.head())

        self.temporal_events_df = self.get_long_temporal_events_df(major_event = False)
        self.characters_temporal_df = self.get_characters_temporal_events_df(major_event = False)
        self.subj_dobj_link_df = self.get_subj_dobj_link_df()

        self.save_characters_temporal_df_to_csv()
        self.save_subj_dobj_link_df_to_csv()

        if major_actions_args_df is not None:
            self.major_actions_args_df = major_actions_args_df.rename(columns = {'verb_id': 'event_id'})
            self.major_ranks_df = major_ranks_df
            self.add_temporal_ranks_to_actions_args_df(major_event = True)
            self.major_temporal_events_df = self.get_long_temporal_events_df(major_event = True)
            self.major_characters_temporal_df = self.get_characters_temporal_events_df(major_event = True)
            self.save_major_characters_temporal_df_to_csv()


    @classmethod
    def from_files(cls, story_id: str, pipeline_dir, major_event = False):
        story_dir_with_prefix = os.path.join(pipeline_dir, story_id, story_id)
        actions_args_df = pd.read_csv(story_dir_with_prefix + '.actions_args.csv')
        ranks_df = pd.read_csv(story_dir_with_prefix + '.events_temporal_ranks.csv')
        characters_df = pd.read_csv(story_dir_with_prefix + '.character_attributes.csv')
        if major_event:
            major_actions_args_df = pd.read_csv(os.path.join(pipeline_dir, story_id, 'major_actions_args.csv'))
            major_ranks_df = pd.read_csv(story_dir_with_prefix + '.major_events_temporal_ranks.csv')
            return cls(story_id, characters_df, actions_args_df, ranks_df, pipeline_dir, major_actions_args_df, major_ranks_df)
        else:
            return cls(story_id, characters_df, actions_args_df, ranks_df, pipeline_dir)

    def get_clean_characters_df(self, characters_df):
        characters_df = characters_df.drop(columns = ['clustered_names'])
        characters_df = characters_df.rename(columns = {'coref_idx': 'coref_id', 'easy_name': 'name', 'total': 'total_mentions'})
        return characters_df[['coref_id', 'name', 'name_mentions', 'pronoun_mentions', 'total_mentions', 'gender', 'gender_certainty', 'importance']]

    def add_temporal_ranks_to_actions_args_df(self, major_event):
        if major_event:
            self.major_actions_args_df = self.major_actions_args_df.merge(self.major_ranks_df, on = ['sentence_id', 'event_id'], how = 'inner')
        else:
            self.actions_args_df = self.actions_args_df.merge(self.ranks_df, on = ['sentence_id', 'event_id'], how = 'inner')

    def get_long_temporal_events_df(self, major_event):
        if major_event:
            df = self.major_actions_args_df
        else:
            df = self.actions_args_df 

        wide_dicts = df.to_dict('records')

        events_subj_dicts = []
        events_dobj_dicts = []

        for row in wide_dicts:
            if isinstance(row['subj_coref_ids'], str) and not pd.isna(row['subj_coref_ids']) and row['subj_coref_ids'] != '':
                subj_coref_ids = row['subj_coref_ids'].split(',')
                for coref_id in subj_coref_ids:
                    if coref_id != '':
                        new_row = row.copy()
                        new_row['subj_coref_ids'] = int(coref_id)
                        events_subj_dicts.append(new_row)
            elif not pd.isna(row['subj_coref_ids']) and row['subj_coref_ids'] != '':
                subj_coref_ids = int(row['subj_coref_ids'])
                events_subj_dicts.append(row)

            if not pd.isna(row['dobj_coref_ids']) and row['dobj_coref_ids'] != '' and isinstance(row['dobj_coref_ids'], str):
                dobj_coref_ids = row['dobj_coref_ids'].split(',')
                for coref_id in dobj_coref_ids:
                    if coref_id != '':
                        new_row = row.copy()
                        new_row['dobj_coref_ids'] = int(coref_id)
                        events_dobj_dicts.append(new_row)
            elif not pd.isna(row['dobj_coref_ids']) and row['dobj_coref_ids'] != '':
                dobj_coref_ids = int(row['dobj_coref_ids'])
                events_dobj_dicts.append(row)

        events_subj_df = pd.DataFrame(events_subj_dicts).drop(columns = ['dobj_coref_ids', 'dobj_start_byte', 'dobj_end_byte', 'dobj_start_byte_text', 'dobj_end_byte_text', 'event_label'])
        events_subj_df.rename(columns = {'subj_coref_ids': 'coref_id', 
                                         'subj_start_byte': 'arg_start_byte_sentence', 
                                         'subj_end_byte': 'arg_end_byte_sentence', 
                                         'subj_start_byte_text': 'arg_start_byte_text', 
                                         'subj_end_byte_text': 'arg_end_byte_text',
                                         'supersense_category_x': 'supersense_category'}, inplace = True)
        events_subj_df['verb'] = events_subj_df['verb'].str.lower()
        events_subj_df['argument'] = 'subject'

        events_dobj_df = pd.DataFrame(events_dobj_dicts).drop(columns = ['subj_coref_ids', 'subj_start_byte', 'subj_end_byte', 'subj_start_byte_text', 'subj_end_byte_text', 'event_label'])
        events_dobj_df.rename(columns = {'dobj_coref_ids': 'coref_id', 
                                         'dobj_start_byte': 'arg_start_byte_sentence', 
                                         'dobj_end_byte': 'arg_end_byte_sentence', 
                                         'dobj_start_byte_text': 'arg_start_byte_text', 
                                         'dobj_end_byte_text': 'arg_end_byte_text',
                                         'supersense_category_x': 'supersense_category'}, inplace = True)
        events_dobj_df['verb'] = events_dobj_df['verb'].str.lower()
        events_dobj_df['argument'] = 'direct_object'

        temporal_events_df = pd.concat([events_subj_df, events_dobj_df], axis = 0)
        temporal_events_df = temporal_events_df.rename(columns = {'verb': 'event'})

        return temporal_events_df

    def get_characters_temporal_events_df(self, major_event):
        if major_event:
            df = self.major_temporal_events_df
        else:
            df = self.temporal_events_df
        characters_temporal_df = df.merge(self.characters_df, on = 'coref_id', how = 'left')
        return characters_temporal_df

    def get_subj_dobj_link_df(self):
        args_df_columns = ['coref_id', 'argument', 'sentence_id', 'event_id', 'event', 'supersense_category']
        subj_df = self.characters_temporal_df[self.characters_temporal_df['argument'] == 'subject'][args_df_columns]
        dobj_df = self.characters_temporal_df[self.characters_temporal_df['argument'] == 'direct_object'][args_df_columns]

        subj_dobj_link_df = subj_df.merge(dobj_df, on = ['sentence_id', 'event_id'], how = 'inner')
        subj_dobj_link_df.drop(columns = ['argument_x', 'argument_y', 'event_y', 'supersense_category_y'], inplace = True)
        subj_dobj_link_df.rename(columns = {'coref_id_x': 'coref_id_subj', 'coref_id_y': 'coref_id_dobj', 'event_x': 'event', 'supersense_category_x': 'supersense_category'}, inplace = True)
        return subj_dobj_link_df[['sentence_id', 'event_id', 'event', 'supersense_category', 'coref_id_subj', 'coref_id_dobj']]

    def save_characters_temporal_df_to_csv(self):
        characters_temporal_file = os.path.join(self.pipeline_dir, self.story_id, self.story_id) + '.characters_temporal_events.csv'
        self.characters_temporal_df.to_csv(characters_temporal_file, index = False)
        print('Saving {} characters_temporal_events CSV to: {}'.format(self.story_id, characters_temporal_file))

    def save_subj_dobj_link_df_to_csv(self):
        subj_dobj_link_file = os.path.join(self.pipeline_dir, self.story_id, self.story_id) + '.characters_subj_dobj_link.csv'
        self.subj_dobj_link_df.to_csv(subj_dobj_link_file, index = False)
        print('Saving {} characters_subj_dobj_link CSV to: {}'.format(self.story_id, subj_dobj_link_file))

    def save_major_characters_temporal_df_to_csv(self):
        major_characters_temporal_file = os.path.join(self.pipeline_dir, self.story_id, self.story_id) + '.major_characters_temporal_events.csv'
        self.major_characters_temporal_df.to_csv(major_characters_temporal_file, index = False)
        print('Saving {} major_characters_temporal_events CSV to: {}'.format(self.story_id, major_characters_temporal_file))



