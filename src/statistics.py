import csv
import json
from numbers import Number
from operator import itemgetter
import os 
import re
from typing import List

import numpy as np
from nltk.stem import WordNetLemmatizer
import pandas as pd

from src.numpy_encoder import NumpyEncoder

class Statistics:
    def __init__(self, story_id: str, characters_df: pd.DataFrame, 
                 characters_temporal_df: pd.DataFrame, sentences_df: pd.DataFrame,
                 pipeline_dir: str, major_characters_temporal_df: None):
        self.story_id = story_id
        self.pipeline_dir = pipeline_dir

        characters_df.rename(columns = {'coref_idx': 'coref_id'}, inplace = True)
        characters_temporal_df = self.set_event_lemmas(characters_temporal_df)
        if major_characters_temporal_df is not None:
            major_characters_temporal_df = self.set_event_lemmas(major_characters_temporal_df)

        self.character_statistics = self.get_character_statistics_json(characters_df, characters_temporal_df, sentences_df)
        self.event_statistics, subj_gender_odds, dobj_gender_odds = self.get_event_statistics_json(characters_df, characters_temporal_df, sentences_df)
        self.story_statistics = self.get_story_statistics_json(characters_df, characters_temporal_df, sentences_df, subj_gender_odds, dobj_gender_odds)

        self.save_json('characters', major_event = False)
        self.save_json('events', major_event = False)
        self.save_json('story', major_event = False)

        if major_characters_temporal_df is not None: 
            self.major_character_statistics = self.get_character_statistics_json(characters_df, major_characters_temporal_df, sentences_df)
            self.major_event_statistics, subj_gender_odds, dobj_gender_odds = self.get_event_statistics_json(characters_df, major_characters_temporal_df, sentences_df)
            self.major_story_statistics = self.get_story_statistics_json(characters_df, major_characters_temporal_df, sentences_df, subj_gender_odds, dobj_gender_odds)

            self.save_json('characters', major_event = True)
            self.save_json('events', major_event = True)
            self.save_json('story', major_event = True)


    @classmethod
    def from_files(cls, story_id: str, pipeline_dir: str, major_event = False):
        story_dir_with_prefix = os.path.join(pipeline_dir, story_id)
        sentences_df = pd.read_csv(os.path.join(story_dir_with_prefix, story_id + '.sentences.csv'))
        characters_df = pd.read_csv(os.path.join(story_dir_with_prefix, story_id + '.character_attributes.csv'))
        characters_temporal_df = pd.read_csv(os.path.join(story_dir_with_prefix, story_id + '.characters_temporal_events.csv'))
        if major_event:
            major_characters_temporal_df = pd.read_csv(os.path.join(story_dir_with_prefix, story_id + '.major_characters_temporal_events.csv'))
        else:
            major_characters_temporal_df = None
        return cls(story_id, characters_df, characters_temporal_df, sentences_df, pipeline_dir, major_characters_temporal_df)

    def set_event_lemmas(self, characters_temporal_df: pd.DataFrame):
        lemmatizer = WordNetLemmatizer()
        characters_temporal_df['event_lemma'] = characters_temporal_df['event'].apply(lemmatizer.lemmatize, args = ['v'])
        return characters_temporal_df

    def get_character_statistics_json(self, characters_df: pd.DataFrame, characters_temporal_df: pd.DataFrame, sentences_df: pd.DataFrame):
        char_event_counts = pd.DataFrame(characters_temporal_df['coref_id'].value_counts()).reset_index().rename(columns = {'coref_id': 'event_n', 'index': 'coref_id'})
        
        # Arguments
        char_arg_counts = pd.DataFrame(characters_temporal_df.groupby('coref_id')['argument'].value_counts()).rename(columns = {'argument': 'arg_n'}).reset_index()
        char_arg_counts = pd.pivot(char_arg_counts, index = ['coref_id'], columns = ['argument'], values = ['arg_n'])
        char_arg_counts.columns = char_arg_counts.columns.to_flat_index()
        char_arg_counts = char_arg_counts.rename(columns = {('arg_n', 'subject'): 'subject_n', ('arg_n', 'direct_object'): 'direct_object_n'}).reset_index().fillna(0)
        
        # Supersense
        char_supersense_counts = pd.DataFrame(characters_temporal_df.groupby('coref_id')['supersense_category'].value_counts()).rename(columns = {'supersense_category': 'supersense_total_n'}).reset_index()
        char_supersense_counts = pd.pivot(char_supersense_counts, index = ['coref_id'], columns = ['supersense_category'], values = ['supersense_total_n'])
        char_supersense_counts.columns = char_supersense_counts.columns.to_flat_index()

        supersense_rename = {('supersense_total_n', 'verb.body'): 'supersense_body_total_n',
                             ('supersense_total_n', 'verb.change'): 'supersense_change_total_n',
                             ('supersense_total_n', 'verb.cognition'): 'supersense_cognition_total_n',
                             ('supersense_total_n', 'verb.communication'): 'supersense_communication_total_n',
                             ('supersense_total_n', 'verb.competition'): 'supersense_competition_total_n',
                             ('supersense_total_n', 'verb.consumption'): 'supersense_consumption_total_n',
                             ('supersense_total_n', 'verb.contact'): 'supersense_contact_total_n',
                             ('supersense_total_n', 'verb.creation'): 'supersense_creation_total_n',
                             ('supersense_total_n', 'verb.emotion'): 'supersense_emotion_total_n',
                             ('supersense_total_n', 'verb.motion'): 'supersense_motion_total_n',
                             ('supersense_total_n', 'verb.perception'): 'supersense_perception_total_n',
                             ('supersense_total_n', 'verb.possession'): 'supersense_possession_total_n',
                             ('supersense_total_n', 'verb.social'): 'supersense_social_total_n',
                             ('supersense_total_n', 'verb.weather'): 'supersense_social_total_n'

                            }
        char_supersense_counts = char_supersense_counts.rename(columns = supersense_rename).reset_index().fillna(0)
        char_supersense_arg_counts = pd.DataFrame(characters_temporal_df.groupby(['coref_id', 'argument'])['supersense_category'].value_counts()).rename(columns = {'supersense_category': 'supersense_total_n'}).reset_index()
        char_supersense_arg_counts = pd.pivot(data = char_supersense_arg_counts, index = ['coref_id'], columns = ['argument', 'supersense_category'], values = ['supersense_total_n'])
        char_supersense_arg_counts.columns = char_supersense_arg_counts.columns.to_flat_index()
        supersense_arg_rename = {('supersense_total_n', 'subject', 'verb.communication'): 'supersense_communication_subj_n',
                                 ('supersense_total_n', 'direct_object', 'verb.communication'): 'supersense_communication_dobj_n',
                                 ('supersense_total_n', 'subject', 'verb.motion'): 'supersense_motion_subj_n',
                                 ('supersense_total_n', 'direct_object', 'verb.motion'): 'supersense_motion_dobj_n',
                                 ('supersense_total_n', 'subject', 'verb.cognition'): 'supersense_cognition_subj_n',
                                 ('supersense_total_n', 'direct_object', 'verb.cognition'): 'supersense_cognition_dobj_n',
                                 ('supersense_total_n', 'subject', 'verb.social'): 'supersense_social_subj_n',
                                 ('supersense_total_n', 'direct_object', 'verb.social'): 'supersense_social_dobj_n',
                                 ('supersense_total_n', 'subject', 'verb.perception'): 'supersense_perception_subj_n',
                                 ('supersense_total_n', 'direct_object', 'verb.perception'): 'supersense_perception_dobj_n',
                                 ('supersense_total_n', 'subject', 'verb.contact'): 'supersense_contact_subj_n',
                                 ('supersense_total_n', 'direct_object', 'verb.contact'): 'supersense_contact_dobj_n',
                                 ('supersense_total_n', 'subject', 'verb.possession'): 'supersense_possession_subj_n',
                                 ('supersense_total_n', 'direct_object', 'verb.possession'): 'supersense_possession_dobj_n',
                                 ('supersense_total_n', 'direct_object', 'verb.change'): 'supersense_change_dobj_n',
                                 ('supersense_total_n', 'subject', 'verb.change'): 'supersense_change_subj_n',
                                 ('supersense_total_n', 'subject', 'verb.emotion'): 'supersense_emotion_subj_n',
                                 ('supersense_total_n', 'direct_object', 'verb.emotion'): 'supersense_emotion_dobj_n',
                                 ('supersense_total_n', 'subject', 'verb.creation'): 'supersense_creation_subj_n',
                                 ('supersense_total_n', 'direct_object', 'verb.creation'): 'supersense_creation_dobj_n',
                                 ('supersense_total_n', 'subject', 'verb.body'): 'supersense_body_subj_n',
                                 ('supersense_total_n', 'direct_object', 'verb.body'): 'supersense_body_dobj_n',
                                 ('supersense_total_n', 'subject', 'verb.competition'): 'supersense_competition_subj_n',
                                 ('supersense_total_n', 'direct_object', 'verb.competition'): 'supersense_competition_dobj_n',
                                 ('supersense_total_n', 'subject', 'verb.consumption'): 'supersense_consumption_subj_n',
                                 ('supersense_total_n', 'direct_object', 'verb.consumption'): 'supersense_consumption_dobj_n',
                                 ('supersense_total_n', 'subject', 'verb.weather'): 'supersense_weather_total_n',
                                 ('supersense_total_n', 'direct_object', 'verb.weather'): 'supersense_weather_total_n'
                                }
        char_supersense_arg_counts = char_supersense_arg_counts.rename(columns = supersense_arg_rename).reset_index().fillna(0)

        # Merge
        count_dfs = [char_event_counts, char_arg_counts, char_supersense_counts, char_supersense_arg_counts]

        for i, df in enumerate(count_dfs):
            if i == 0:
                characters_stats_df = characters_df.merge(df, on = 'coref_id', how = 'left')
            else:
                characters_stats_df = characters_stats_df.merge(df, on = 'coref_id', how = 'left')
        characters_stats_df = characters_stats_df.fillna(0)
        character_stats_json = characters_stats_df.set_index('coref_id').to_dict('index')

        for coref_id in character_stats_json.keys():
            char_events_df = characters_temporal_df[characters_temporal_df['coref_id'] == coref_id]
            character_stats_json[coref_id]['event_counts_total'] = char_events_df['event_lemma'].value_counts().to_dict()
            character_stats_json[coref_id]['event_counts_subj'] = char_events_df[char_events_df['argument'] == 'subject']['event_lemma'].value_counts().to_dict()
            character_stats_json[coref_id]['event_counts_dobj'] = char_events_df[char_events_df['argument'] == 'direct_object']['event_lemma'].value_counts().to_dict()

        return character_stats_json

    def get_event_statistics_json(self, characters_df: pd.DataFrame, characters_temporal_df: pd.DataFrame, sentences_df: pd.DataFrame):
        event_counts = pd.DataFrame(characters_temporal_df['event_lemma'].value_counts()).reset_index().rename(columns = {'event_lemma': 'event_n', 'index': 'event_lemma'})

        # Gender
        event_gender_counts = pd.crosstab(characters_temporal_df['event_lemma'], characters_temporal_df['gender'])
        event_gender_counts = event_gender_counts.reset_index()
        event_gender_counts.columns.name = None
        rename_columns = {'female': 'female_event_n',
                          'male': 'male_event_n',
                          'group/nonbinary': 'group/nonbinary_event_n',
                          'unknown': 'unknown_event_n'}
        event_gender_counts.rename(columns = rename_columns, inplace = True)

        # Argument
        event_arg_counts = pd.crosstab(characters_temporal_df['event_lemma'], characters_temporal_df['argument']).reset_index()
        event_arg_counts.columns.name = None
        rename_columns = {'direct_object': 'dobj_event_n',
                          'subject': 'subj_event_n'}
        event_arg_counts.rename(columns = rename_columns, inplace = True)

        # Argument and Gender
        event_arg_gender_counts = pd.crosstab(characters_temporal_df['event_lemma'], [characters_temporal_df['argument'], characters_temporal_df['gender']])
        event_arg_gender_counts.columns = event_arg_gender_counts.columns.to_flat_index()
        rename_columns = {('direct_object', 'female'): 'dobj_female_event_n',
                          ('direct_object', 'group/nonbinary'): 'dobj_group_nb_event_n',
                          ('direct_object', 'male'): 'dobj_male_event_n',
                          ('direct_object', 'unknown'): 'dobj_unknown_event_n',
                          ('subject', 'female'): 'subj_female_event_n',
                          ('subject', 'group/nonbinary'): 'subj_group_nb_event_n',
                          ('subject', 'male'): 'subj_male_event_n',
                          ('subject', 'unknown'): 'subj_unknown_event_n'
                         }
        event_arg_gender_counts.rename(columns = rename_columns, inplace = True)
        event_arg_gender_counts.reset_index(inplace = True)

        # Odds Ratio
        ## Gender
        if 'female_event_n' in event_gender_counts.columns:
            event_female_dict_raw = event_gender_counts[['event_lemma', 'female_event_n']].to_dict('records')
            event_female_dict = {}
            for row in event_female_dict_raw:
                event_female_dict[row['event_lemma']] = row['female_event_n']

        if 'male_event_n' in event_gender_counts.columns:
            event_male_dict_raw = event_gender_counts[['event_lemma', 'male_event_n']].to_dict('records')
            event_male_dict = {}
            for row in event_male_dict_raw:
                event_male_dict[row['event_lemma']] = row['male_event_n']

        if 'female_event_n' in event_gender_counts.columns and 'male_event_n' in event_gender_counts.columns:
            gender_odds = pd.DataFrame.from_dict(odds_ratio(event_female_dict, event_male_dict), orient = 'index').reset_index().rename(columns = {'index': 'event_lemma', 0: 'male_female_odds'})
            if len(gender_odds) == 0:
                gender_odds = None
        else:
            gender_odds = None

        ### Subject
        if 'subj_female_event_n' in event_arg_gender_counts.columns:
            event_female_subj_dict_raw = event_arg_gender_counts[['event_lemma', 'subj_female_event_n']].to_dict('records')
            event_female_subj_dict = {}
            for row in event_female_subj_dict_raw:
                event_female_subj_dict[row['event_lemma']] = row['subj_female_event_n']

        if 'subj_male_event_n' in event_arg_gender_counts.columns:
            event_male_subj_dict_raw = event_arg_gender_counts[['event_lemma', 'subj_male_event_n']].to_dict('records')
            event_male_subj_dict = {}
            for row in event_male_subj_dict_raw:
                event_male_subj_dict[row['event_lemma']] = row['subj_male_event_n']

        if 'subj_female_event_n' in event_arg_gender_counts.columns and 'subj_male_event_n' in event_arg_gender_counts.columns:
            subj_gender_odds = pd.DataFrame.from_dict(odds_ratio(event_female_subj_dict, event_male_subj_dict), orient = 'index').reset_index().rename(columns = {'index': 'event_lemma', 0: 'male_female_subj_odds'})
            if len(subj_gender_odds) == 0:
                subj_gender_odds = None
        else:
            subj_gender_odds = None

        ### Direct Object
        if 'dobj_female_event_n' in event_arg_gender_counts.columns:
            event_female_dobj_dict_raw = event_arg_gender_counts[['event_lemma', 'dobj_female_event_n']].to_dict('records')
            event_female_dobj_dict = {}
            for row in event_female_dobj_dict_raw:
                event_female_dobj_dict[row['event_lemma']] = row['dobj_female_event_n']
        
        if 'dobj_male_event_n' in event_arg_gender_counts.columns:
            event_male_dobj_dict_raw = event_arg_gender_counts[['event_lemma', 'dobj_male_event_n']].to_dict('records')
            event_male_dobj_dict = {}
            for row in event_male_dobj_dict_raw:
                event_male_dobj_dict[row['event_lemma']] = row['dobj_male_event_n']
           
        if 'dobj_female_event_n' in event_arg_gender_counts.columns and 'dobj_male_event_n' in event_arg_gender_counts.columns: 
            dobj_gender_odds = pd.DataFrame.from_dict(odds_ratio(event_female_dobj_dict, event_male_dobj_dict), orient = 'index').reset_index().rename(columns = {'index': 'event_lemma', 0: 'male_female_dobj_odds'})
            if len(dobj_gender_odds) == 0:
                dobj_gender_odds = None
        else:
            dobj_gender_odds = None

        ## Importance
        event_importance_counts = pd.crosstab(characters_temporal_df['event_lemma'], characters_temporal_df['importance']).reset_index()
        event_importance_counts
        event_importance_counts.columns = event_importance_counts.columns.to_flat_index()
        event_importance_counts.columns.name = None
        rename_columns = {'primary': 'primary_event_n',
                          'secondary': 'secondary_event_n',
                          'tertiary': 'tertiary_event_n'}
        event_importance_counts.rename(columns = rename_columns, inplace = True)

        if 'primary_event_n' in event_importance_counts.columns:
            event_primary_dict_raw = event_importance_counts[['event_lemma', 'primary_event_n']].to_dict('records')
            event_primary_dict = {}
            for row in event_primary_dict_raw:
                event_primary_dict[row['event_lemma']] = row['primary_event_n']
        
        if 'secondary_event_n' in event_importance_counts.columns:
            event_secondary_dict_raw = event_importance_counts[['event_lemma', 'secondary_event_n']].to_dict('records')
            event_secondary_dict = {}
            for row in event_secondary_dict_raw:
                event_secondary_dict[row['event_lemma']] = row['secondary_event_n']
            
        if 'tertiary_event_n' in event_importance_counts.columns:
            event_tertiary_dict_raw = event_importance_counts[['event_lemma', 'tertiary_event_n']].to_dict('records')
            event_tertiary_dict = {}
            for row in event_tertiary_dict_raw:
                event_tertiary_dict[row['event_lemma']] = row['tertiary_event_n']

        if 'primary_event_n' in event_importance_counts.columns and 'secondary_event_n' in event_importance_counts.columns:
            importance_pri_sec_odds = pd.DataFrame.from_dict(odds_ratio(event_secondary_dict, event_primary_dict), orient = 'index').reset_index().rename(columns = {'index': 'event_lemma', 0: 'primary_secondary_odds'})
        else: 
            importance_pri_sec_odds = None
        if 'primary_event_n' in event_importance_counts.columns and 'tertiary_event_n' in event_importance_counts.columns:
            importance_pri_ter_odds = pd.DataFrame.from_dict(odds_ratio(event_tertiary_dict, event_primary_dict), orient = 'index').reset_index().rename(columns = {'index': 'event_lemma', 0: 'primary_tertiary_odds'})
        else:
            importance_pri_ter_odds = None

        # Merge
        count_dfs = [event_counts, event_gender_counts, event_arg_counts, event_arg_gender_counts, gender_odds, subj_gender_odds, dobj_gender_odds, event_importance_counts, importance_pri_sec_odds, importance_pri_ter_odds]
        events_df = characters_temporal_df[['event_lemma', 'supersense_category']].drop_duplicates('event_lemma')
        for i, df in enumerate(count_dfs):
            if df is not None:
                if i == 0:
                    events_stats_df = events_df.merge(df, on = 'event_lemma', how = 'left')
                else:
                    events_stats_df = events_stats_df.merge(df, on = 'event_lemma', how = 'left')

        event_stats_json = events_stats_df.set_index('event_lemma').to_dict('index')
        for event_lemma in event_stats_json.keys():
            event_stats_json[event_lemma]['event_occurances'] = {}
            event_lemma_df = characters_temporal_df[characters_temporal_df['event_lemma'] == event_lemma][['sentence_id', 'event_id', 'temporal_rank', 'coref_id', 'argument', 'gender', 'importance']]
            event_ids = pd.Series([tuple(r) for r in event_lemma_df[['sentence_id', 'event_id']].values]).unique()
            for sentence_id, event_id in event_ids:
                story_position = round(sentence_id/len(sentences_df), 2)
                event_i_df = event_lemma_df[(event_lemma_df['sentence_id'] == sentence_id) & (event_lemma_df['event_id'] == event_id)][['temporal_rank', 'coref_id', 'argument', 'gender', 'importance']]
                event_stats_json[event_lemma]['event_occurances'][str(sentence_id) + ':' + str(event_id) + ':' + str(story_position)] = event_i_df.to_dict('records')

        return event_stats_json, subj_gender_odds, dobj_gender_odds

    def get_story_statistics_json(self, characters_df: pd.DataFrame, characters_temporal_df: pd.DataFrame, sentences_df: pd.DataFrame,
                                  subj_gender_odds: pd.DataFrame, dobj_gender_odds: pd.DataFrame):
        story_stats_json = {}

        # Character
        story_stats_json['characters'] = {}
        story_stats_json['characters']['n'] = len(characters_df)

        ## Gender
        for gender in ['female', 'male', 'group/nonbinary', 'unknown']:
            story_stats_json['characters'][gender] = {}
            try:
                story_stats_json['characters'][gender]['n'] = characters_df['gender'].value_counts()[gender]
                story_stats_json['characters'][gender]['p'] = round(characters_df['gender'].value_counts()[gender]/len(characters_df), 2)
            except KeyError:
                story_stats_json['characters'][gender]['n'] = 0
                story_stats_json['characters'][gender]['p'] = 0

        ## Importance
        for importance in ['primary', 'secondary', 'tertiary']:
            story_stats_json['characters'][importance] = {}
            try: 
                story_stats_json['characters'][importance]['n'] = characters_df['importance'].value_counts()[importance]
                story_stats_json['characters'][importance]['p'] = round(characters_df['importance'].value_counts()[importance]/len(characters_df), 2)
            except KeyError:
                story_stats_json['characters'][importance]['n'] = 0 
                story_stats_json['characters'][importance]['p'] = 0 

        ## Gender x Importance
        char_gender_imp_counts = pd.crosstab(characters_df['gender'], characters_df['importance']).reset_index()
        char_gender_imp_counts.columns.name = None

        char_gender_imp_p = pd.crosstab(characters_df['gender'], characters_df['importance'], normalize = True).reset_index()
        char_gender_imp_p.columns.name = None

        for gender in ['female', 'male', 'group/nonbinary', 'unknown']:
            for importance in ['primary', 'secondary', 'tertiary']:
                gender_importance = gender + '_' + importance
                story_stats_json['characters'][gender_importance] = {}
                try: 
                    story_stats_json['characters'][gender_importance]['n'] = char_gender_imp_counts[char_gender_imp_counts['gender'] == gender][importance].values[0]
                    story_stats_json['characters'][gender_importance]['p'] = round(char_gender_imp_p[char_gender_imp_p['gender'] == gender][importance].values[0], 2)
                except KeyError:
                    story_stats_json['characters'][gender_importance]['n'] = 0 
                    story_stats_json['characters'][gender_importance]['p'] = 0
                except IndexError:
                    story_stats_json['characters'][gender_importance]['n'] = 0 
                    story_stats_json['characters'][gender_importance]['p'] = 0

        # Appearances
        story_stats_json['characters']['appearances'] = {}
        story_stats_json['characters']['appearances']['all'] = characters_df['total'].tolist()
        for gender in ['female', 'male', 'group/nonbinary', 'unknown']:
            story_stats_json['characters']['appearances'][gender] = characters_df[characters_df['gender'] == gender]['total'].tolist()

        # Events
        story_stats_json['events'] = {}

        story_event_counts = pd.crosstab([characters_temporal_df['gender'], characters_temporal_df['importance']], characters_temporal_df['argument'], margins = True, margins_name = 'total').reset_index()
        story_event_gender_counts = pd.crosstab(characters_temporal_df['gender'], characters_temporal_df['argument'], margins = True, margins_name = 'total').reset_index()
        story_event_counts.columns.name = None
        story_event_gender_counts.drop(story_event_gender_counts.tail(1).index, inplace = True)
        story_event_gender_counts['importance'] = 'total'
        story_event_gender_counts.columns.name = None
        story_event_counts = pd.concat([story_event_counts, story_event_gender_counts]).sort_values(by = 'gender')
        story_stats_json['events']['counts'] = story_event_counts.to_dict('records')

        ## Top Events
        story_stats_json['events']['top_events'] = {}

        if subj_gender_odds is not None:
            subj_gender_odds['argument'] = 'subject'
            subj_gender_odds.rename(columns = {'male_female_subj_odds': 'male_female_odds'}, inplace = True)

        if dobj_gender_odds is not None:
            dobj_gender_odds['argument'] = 'direct_object'
            dobj_gender_odds.rename(columns = {'male_female_dobj_odds': 'male_female_odds'}, inplace = True)

        if subj_gender_odds is not None and dobj_gender_odds is not None:
            gender_odds_top = pd.concat([subj_gender_odds, dobj_gender_odds]).sort_values(by = 'male_female_odds', ascending = False)
        elif subj_gender_odds is not None and dobj_gender_odds is None:
            gender_odds_top = subj_gender_odds.sort_values(by = 'male_female_odds', ascending = False)
        elif dobj_gender_odds is not None and subj_gender_odds is None:
            gender_odds_top = dobj_gender_odds.sort_values(by = 'male_female_odds', ascending = False)
        else:
            gender_odds_top = None

        if gender_odds_top is not None:
            gender_odds_top_male = gender_odds_top.copy()
            gender_odds_top_female = gender_odds_top.copy()

        try: 
            gender_odds_top_female['female_male_odds'] = round(1/gender_odds_top_female['male_female_odds'], 2)
            gender_odds_top_female.drop(columns = 'male_female_odds', inplace = True)
            gender_odds_top_female.sort_values(by = 'female_male_odds', ascending = False, inplace = True)
            gender_odds_top_male = gender_odds_top_male[gender_odds_top_male['male_female_odds'] >= 2].to_dict('records')
            gender_odds_top_female = gender_odds_top_female[gender_odds_top_female['female_male_odds'] >= 2].to_dict('records')

            story_stats_json['events']['top_events'] = {}
            story_stats_json['events']['top_events']['female'] = gender_odds_top_female
            story_stats_json['events']['top_events']['male'] = gender_odds_top_male

            story_stats_json['events']['top_events']['unbiased'] = gender_odds_top[(gender_odds_top['male_female_odds'] < 2) & (gender_odds_top['male_female_odds'] > 0.5)].to_dict('records')
        except:
            print("Could not calculate top events by odds ratio because at least one gender group has no associated events.")

        return story_stats_json

    def save_json(self, name_json: str, major_event: bool):
        if name_json == 'characters':
            if major_event:
                stats_json = self.major_character_statistics
            else:
                stats_json = self.character_statistics
        elif name_json == 'events':
            if major_event:
                stats_json = self.major_event_statistics
            else:
                stats_json = self.event_statistics
        elif name_json == 'story': 
            if major_event:
                stats_json = self.major_story_statistics
            else:
                stats_json = self.story_statistics

        nan_to_null_json(stats_json)

        if major_event:
            json_path = os.path.join(self.pipeline_dir, self.story_id, self.story_id + '.' + name_json + '_major_statistics' + '.json')
        else: 
            json_path = os.path.join(self.pipeline_dir, self.story_id, self.story_id + '.' + name_json + '_statistics' + '.json')

        with open(json_path, 'w') as json_file:
            json.dump(stats_json, json_file, indent = 4, cls = NumpyEncoder)
        print('Saving {} {} statistics CSV to: {}'.format(self.story_id, name_json, json_path))

def odds_ratio(f_dict, m_dict, threshold = 2, haldane_correction = True):
    very_small_value = 0.00001
    if len(f_dict.keys()) != len(m_dict.keys()):
        raise Exception("The category for analyzing the male and female should be the same!")
    else:
        odds_ratio = {}
        total_num_f = sum(f_dict.values())
        total_num_m = sum(m_dict.values())
        for key in f_dict.keys():
            if haldane_correction == True:
                m_num = m_dict[key] + 0.5
                f_num = f_dict[key] + 0.5
            elif haldane_correction == False:
                m_num = m_dict[key]
                f_num = f_dict[key]             
            non_f_num = total_num_f - f_num
            non_m_num = total_num_m - m_num
            if haldane_correction == True:
                if f_num >= threshold or m_num >= threshold:
                    odds_ratio[key] = round((m_num / f_num) / (non_m_num / non_f_num), 2)
                else:
                    continue
            if haldane_correction == False:
                # we only consider the events where there are at least {thresohld} occurences for both gender
                if f_num >= threshold and m_num >= threshold:
                    odds_ratio[key] = round((m_num / f_num) / (non_m_num / non_f_num), 2)
                else:
                    continue

        return dict(sorted(odds_ratio.items(), key=itemgetter(1), reverse = True))

def nan_to_null_json(d):
    for k,v in d.items():        
        if isinstance(v, dict):
            nan_to_null_json(v)
        else:            
            if isinstance(v, Number) and np.isnan(v):
                d[k] = None
