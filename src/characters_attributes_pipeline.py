# Standard Library
import json
from typing import List
import re
import os

# Third Party
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import numpy as np 
import pandas as pd

class CharacterAttributesPipeline:
    def __init__(self, story_id: str, pipeline_dir: str, 
                 sentences_df: pd.DataFrame, sentences_characters_df: pd.DataFrame, characters_df: pd.DataFrame,
                 attributes = ['gender']):
        self.story_id = story_id 
        self.pipeline_dir = pipeline_dir 
        self.attributes = attributes
        self.lemmatizer = WordNetLemmatizer()

        self.sentences_df = sentences_df
        self.sentences_characters_df = sentences_characters_df
        self.characters_df = characters_df
        self.characters =  characters_df['easy_name'].unique()

        self.male_pronouns = {"he", "him", "his"}
        self.female_pronouns = {"she", "her", "hers"}
        self.group_nonbinary_pronouns = {"they", "them", "their", "theirs"}
        self.male_nouns = {"man", "men", "gentleman", "gentlemen", "boy", "husband", "father", "brother", "grandfather", "nephew", "emperor", "king", "prince"}
        self.female_nouns = {"woman", "women", "lady", "girl", "maiden", "mother", "wife", "sister", "grandmother", "niece", "emperess", "queen", "princess"}

        if 'gender' in self.attributes:
            self.set_gender_in_characters_df()
        if 'importance' in self.attributes:
            self.set_character_importance_in_characters_df()
        if 'age' in self.attributes:
            self.set_age_in_characters_df()

        self.save_characters_df_with_attributes()

    @classmethod
    def from_files(cls, story_id: str, pipeline_dir: str, attributes = ['gender']):
        story_dir_with_prefix = os.path.join(pipeline_dir, story_id, story_id)
        sentences_path = story_dir_with_prefix + '.sentences.csv'
        sentences_characters_path = story_dir_with_prefix + '.sentences_characters.csv'
        characters_path = story_dir_with_prefix + '.character_meta_pron'

        sentences_df = pd.read_csv(sentences_path)
        sentences_characters_df = pd.read_csv(sentences_characters_path)
        characters_df = pd.read_csv(characters_path, delimiter = '\t')

        return cls(story_id, pipeline_dir, sentences_df, sentences_characters_df, characters_df, attributes)

    # Character Gender: Female, Male, Mixed Group/Nonbinary
    def get_gender_of_character_name(self, character_name: str):
        gender_flags = []
        tokens = [self.lemmatizer.lemmatize(token.lower()) for token in character_name.split()]
        
        # Remove possessives
        for token in tokens:
            for possessive_flag in ["'", "his", "her", "their"]:
                if possessive_flag == token:
                    tokens.remove(token)

        if not set(tokens).isdisjoint(self.male_nouns):
            gender_flags.append('male')
        if not set(tokens).isdisjoint(self.female_nouns):
            gender_flags.append('female')

        return gender_flags

    def get_gender_of_pronouns(self, corefs: List):
        gender_flags = []
        for pronoun in corefs:
            if pronoun in self.male_pronouns:
                gender_flags.append('male')
            if pronoun in self.female_pronouns:
                gender_flags.append('female')
            if pronoun in self.group_nonbinary_pronouns:
                gender_flags.append('group/nonbinary')
        return gender_flags

    def get_gender_with_certainty(self, name_gender_flags, pronoun_gender_flags):
        if len(pd.Series(name_gender_flags, dtype = 'object').unique()) == 1:
            name_gender = pd.Series(name_gender_flags, dtype = 'object').unique().tolist()[0]
            name_gender_certainty = 1.0
        elif len(pd.Series(name_gender_flags, dtype = 'object').unique()) > 1:
            name_gender = pd.Series(name_gender_flags, dtype = 'object').mode().values[0]
            name_gender_certainty = (pd.Series(name_gender_flags, dtype = 'object').value_counts()/len(name_gender_flags)).to_dict()[name_gender]
        else:
            name_gender = 'unknown'
            name_gender_certainty = 0.0

        if len(pronoun_gender_flags) > 0:
            pronoun_gender = pd.Series(pronoun_gender_flags, dtype = 'object').mode().values[0]
            pronoun_gender_certainty = (pd.Series(pronoun_gender_flags, dtype = 'object').value_counts()/len(pronoun_gender_flags)).to_dict()[pronoun_gender]
        else:
            pronoun_gender = 'unknown'
            pronoun_gender_certainty = 0.0

        gender_flags = list(np.repeat(name_gender_flags, len(pronoun_gender_flags) + 1)) + pronoun_gender_flags

        if name_gender_certainty == 1.0:
            return name_gender, name_gender_certainty
        elif name_gender == 'unknown' and pronoun_gender == 'unknown':
            return name_gender, name_gender_certainty
        else:
            gender = pd.Series(gender_flags, dtype = 'object').mode().tolist()[0]
            gender_certainty = (pd.Series(gender_flags, dtype = 'object').value_counts()/len(gender_flags)).to_dict()[gender]
            return gender, gender_certainty

    def set_gender_in_characters_df(self):
        for coref in self.characters_df['coref_idx'].unique():
            char_df = self.characters_df[self.characters_df['coref_idx'] == coref]
            character_name = char_df['easy_name'].values[0]
            name_gender_flags = self.get_gender_of_character_name(character_name)
            char_sentences_df = self.sentences_characters_df[self.sentences_characters_df['coref_id'] == coref]
            pronoun_gender_flags = self.get_gender_of_pronouns(char_sentences_df['character_token'].tolist())
            gender, certainty = self.get_gender_with_certainty(name_gender_flags, pronoun_gender_flags)
            self.characters_df.loc[self.characters_df['coref_idx'] == coref, 'gender'] = gender
            self.characters_df.loc[self.characters_df['coref_idx'] == coref, 'gender_certainty'] = certainty

    # Character Importance: Primary, Secondary, Tertiary
    def get_most_important_character(self):
        return self.characters_df.loc[self.characters_df['total'].idxmax(), 'coref_idx']

    def set_character_importance_in_characters_df(self):
        most_important_character = self.get_most_important_character()
        most_important_appearances = self.characters_df.loc[self.characters_df['coref_idx'] == most_important_character, 'total']
        secondary_threshold = int(most_important_appearances * 0.66)
        tertiary_threshold = int(most_important_appearances * 0.33)
        self.characters_df['importance'] = self.characters_df['total'].apply(set_character_importance, args = (secondary_threshold, tertiary_threshold))

    # Character Age
    def set_age_in_characters_df(self):
        pass 

    def save_characters_df_with_attributes(self):
        self.characters_df.to_csv(self.pipeline_dir + self.story_id + '/' + self.story_id + '.character_attributes.csv', index = False)

# Pandas Apply Function for Character Importance 
def set_character_importance(appearances, secondary_threshold, tertiary_threshold):
    if appearances >= secondary_threshold:
        return 'primary'
    elif appearances >= tertiary_threshold and appearances < secondary_threshold:
        return 'secondary'
    else:
        return 'tertiary'
   


