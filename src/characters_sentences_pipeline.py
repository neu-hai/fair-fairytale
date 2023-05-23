import os
import re
import csv

from typing import List

import pandas as pd

from src.sentences import split_sentences 
from src.utils import read_story_txt_file

class CharacterSentencePipeline:
    def __init__(self, 
                 story_id: str, 
                 entities_df: pd.DataFrame, 
                 tokens_df: pd.DataFrame, 
                 story_text: str, 
                 pipeline_dir: str,
                 pronoun_only_entities = False):
        self.story_id = story_id

        story_dir_with_prefix = os.path.join(pipeline_dir,story_id, story_id)
        self.output_sentences_file = story_dir_with_prefix + '.sentences.csv'
        self.output_sentences_by_characters_file = story_dir_with_prefix + '.sentences_characters.csv'

        self.story_text = story_text
        self.sentences = split_sentences(story_text)

        self.tokens_df = tokens_df
        self.characters_df = entities_df[entities_df['cat'] == 'PER']

        self.run_pipeline(pronoun_only_entities)
    
    @classmethod
    def from_files(cls, story_id, pipeline_dir, pronoun_only_entities = False):
        story_dir_with_prefix = os.path.join(pipeline_dir,story_id, story_id)
        entities_file = story_dir_with_prefix + '.entities'
        tokens_file = story_dir_with_prefix + '.tokens'
        story_txt_file = story_dir_with_prefix + '.txt'

        entities_df = pd.read_csv(entities_file, sep='\t')
        tokens_df = pd.read_csv(tokens_file, delimiter = '\t')
        story_text = read_story_txt_file(story_txt_file)

        return cls(story_id, entities_df, tokens_df, story_text, pipeline_dir, pronoun_only_entities)

    def save_sentences_csv(self):
        self.sentences_df.to_csv(self.output_sentences_file, index = False)
        print("CSV of sentences saved to path: {}".format(self.output_sentences_file))
        
    def save_sentences_by_characters_csv(self):
        output_sentences_by_characters_file
        self.sentences_by_character_df.to_csv(self.output_sentences_by_characters_file, index = False)
        print("CSV of sentences by character saved to path: {}".format(self.output_sentences_by_characters_file))
        
    def save_output(self):
        self.save_sentences_csv()
        self.save_sentences_by_character_df()
            
    def get_coref_character_names(self, corefs: List[str], pronoun_only_entities: str) -> dict:
        coref_character_dict = {}

        for coref in corefs:        
            all_names = ""

            coref_df = self.characters_df[self.characters_df['COREF'] == coref]

            proper_names = coref_df[coref_df['prop'] == 'PROP']['text'].unique().tolist()

            # Get characters with no proper name
            if len(proper_names) == 0:
                common_names = coref_df[coref_df['prop'] == 'NOM']['text'].unique().tolist()
                # For short text input, allow coref entities to just refer to pronouns. 
                if pronoun_only_entities == True:
                    if len(common_names) == 0:
                        common_names = coref_df[coref_df['prop'] == 'PRON']['text'].unique().tolist()
                if len(common_names) > 1:
                    # Check if just different cases
                    common_name = common_names[0].lower()
                    same_counts = 0
                    for name in common_names:
                        if name.lower() == common_names[0].lower():
                            same_counts += 1
                    if same_counts == len(common_names):
                        all_names = common_names[0].lower()
                    else:
                        all_names = "/".join(common_names)
                if len(common_names) == 1:
                    all_names = common_names[0]
            else:
                all_names = "/".join(proper_names)

            if all_names not in ['', 'anyone', 'everyone', 'no one']:
                coref_character_dict[coref] = all_names

        return coref_character_dict

    @staticmethod
    def get_start_end_bytes_of_sentences(text: str) -> List:
        sentences = split_sentences(text)

        while ('' in sentences):
            sentences.remove('')

        start_end_sentences = []
        for sentence in sentences:
            sentence_start = text.find(sentence)
            sentence_end = sentence_start + len(sentence) + 1
            start_end_sentences.append([text[sentence_start:sentence_end], sentence_start, sentence_end])

        return start_end_sentences

    def get_start_end_bytes_of_coref(self, coref: int) -> List:
        start_end_bytes_of_corefs = []
        char_df = self.corefs_df[self.corefs_df['COREF'] == coref]
        for i, row in char_df.iterrows():
            token_range = list(range(row['start_token'], row['end_token'] + 1))
            char_tokens_df = self.tokens_df[self.tokens_df['token_ID_within_document'].isin(token_range)]
            start_byte = int(char_tokens_df['byte_onset'].tolist()[0])
            end_byte = int(char_tokens_df['byte_offset'].tolist()[-1])
            if (self.story_text[start_byte:end_byte] != row['text']):
                start_end_bytes_of_corefs.append([start_byte, end_byte, self.story_text[start_byte:end_byte]]) 
            else:
                start_end_bytes_of_corefs.append([start_byte, end_byte, row['text']])
        return start_end_bytes_of_corefs

    def get_sentence_end_start_df(self):
        start_end_sentences = self.get_start_end_bytes_of_sentences(self.story_text) 
        return pd.DataFrame(start_end_sentences).reset_index().rename(columns = {'index': 'sentence_id', 0: 'text', 1: 'start', 2: 'end'})

    def get_start_end_bytes_in_sentences(self, coref):
        start_end_coref = self.get_start_end_bytes_of_coref(coref)

        start_end_bytes_in_sentence = []
        
        for (coref_start, coref_end, character_text) in start_end_coref:
            ses_char_df = self.sentences_df[(coref_start >= self.sentences_df['start']) & (coref_end <= self.sentences_df['end'])]
            if len(ses_char_df) > 0:
                sentence_id = ses_char_df['sentence_id'].values[0]
                start_sentence = ses_char_df['start'].values[0]
                end_sentence = ses_char_df['end'].values[0]
                start_coref_in_sentence = coref_start - start_sentence
                end_coref_in_sentence = start_coref_in_sentence + len(character_text)
                character_text_in_sentence = ses_char_df['text'].values[0][start_coref_in_sentence:end_coref_in_sentence]
                start_end_bytes_in_sentence.append([coref, self.coref2character[coref], character_text, sentence_id, start_coref_in_sentence, end_coref_in_sentence])
        return start_end_bytes_in_sentence

    def get_sentences_by_characters_df(self) -> pd.DataFrame:        
        sentences_by_character = []
        for coref in self.corefs:
            sentences_by_character += self.get_start_end_bytes_in_sentences(coref)
        return pd.DataFrame(sentences_by_character).rename(columns = {0: 'coref_id', 
                                                                      1: 'character_name', 
                                                                      2: 'character_token', 
                                                                      3: 'sentence_id', 
                                                                      4: 'start_byte_in_sentence', 
                                                                      5: 'end_byte_in_sentence'})

    @staticmethod
    def check_overlapping_spans(row, spans):
        curr_span = (row['start_byte_in_sentence'], row['end_byte_in_sentence'])
        spans_c = spans.copy()
        spans_c.remove(curr_span)
        for span in spans_c:
            if (curr_span[0] >= span[0]) and (curr_span[1] <= span[1]):
                return 1
        return 0

    def check_if_coref_possessive(self, sentence_character_df: pd.DataFrame):
        if len(sentence_character_df) > 0:
            byte_start_ends_dict = sentence_character_df[['start_byte_in_sentence', 'end_byte_in_sentence']].to_dict('records')
            byte_start_ends = [(row['start_byte_in_sentence'], row['end_byte_in_sentence']) for row in byte_start_ends_dict]
            return sentence_character_df.apply(self.check_overlapping_spans, spans = byte_start_ends, axis = 1) 

    def get_overlapping_coref_series(self):
        overlaps_series_all = []
        for i in range(len(self.sentences)):
            sentence_by_character_df = self.sentences_by_character_df.loc[self.sentences_by_character_df['sentence_id'] == i]
            overlap_series = self.check_if_coref_possessive(sentence_by_character_df)
            overlaps_series_all.append(overlap_series)

        return overlaps_series_all

    def save_sentences_csv(self):
        self.sentences_df.to_csv(self.output_sentences_file, index = False)
        print("Saving {} sentences CSV to: {}".format(self.story_id, self.output_sentences_file))

    def save_characters_by_sentence_csv(self):
        self.sentences_by_character_df.to_csv(self.output_sentences_by_characters_file, index = False)
        print("Saving {} sentences_character CSV to: {}".format(self.story_id, self.output_sentences_by_characters_file))

    def run_pipeline(self, pronoun_only_entities):
        # All corefs
        corefs = self.characters_df['COREF'].unique()

        # All valid corefs (no corefs with only pronouns or possessives)
        self.coref2character = self.get_coref_character_names(corefs, pronoun_only_entities) 
        self.corefs = self.coref2character.keys()

        self.corefs_df = self.characters_df[self.characters_df['COREF'].isin(self.corefs)][['COREF', 'start_token', 'end_token', 'text']].sort_values(by = 'COREF')

        # Output: Sentence DataFrame
        start_end_sentences = self.get_start_end_bytes_of_sentences(self.story_text)
        self.sentences_df = pd.DataFrame(start_end_sentences).reset_index().rename(columns = {'index': 'sentence_id', 0: 'text', 1: 'start', 2: 'end'})

        # Output: Sentences by Character DataFrame
        self.sentences_by_character_df = self.get_sentences_by_characters_df()

        # Get flag for coreferences that overlapping, typically possessives
        coref_overlaps = self.get_overlapping_coref_series()
        self.sentences_by_character_df = self.sentences_by_character_df.merge(pd.concat(coref_overlaps).rename('overlap'), left_index=True, right_index=True)

        self.save_sentences_csv()
        self.save_characters_by_sentence_csv()
