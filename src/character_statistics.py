import os
import re
import csv

from typing import List

import numpy as np
import pandas as pd

from src.sentences import split_sentences 
from src.utils import read_story_txt_file

class CharacterStatistics:
    def __init__(self, story_id: str, characters_df: pd.DataFrame, characters_temporal_df: pd.DataFrame, pipeline_dir: str):
        self.story_id = story_id

    @classmethod
    def from_files(cls, story_id: str, pipeline_dir: str):
        story_dir_with_prefix = pipeline_dir + story_id + '/' + story_id
        characters_df = pd.read_csv(story_dir_with_prefix + '.character_attributes.csv')
        characters_temporal_df = pd.read_csv(story_dir_with_prefix + '.characters_temporal_events.csv')

        return cls(story_id, characters_df, characters_temporal_df, pipeline_dir)

    def calc_event_stats(self, characters_temporal_df: pd.DataFrame):
        characters_stats_df = characters_temporal_df.groupby()


class CharacterStatistics:
    def __init__(self, story_id: str, characters_df: pd.DataFrame, characters_temporal_df: pd.DataFrame, pipeline_dir: str):
        self.story_id = story_id

    @classmethod
    def from_files(cls, story_id: str, pipeline_dir: str):
        story_dir_with_prefix = pipeline_dir + story_id + '/' + story_id
        characters_df = pd.read_csv(story_dir_with_prefix + '.character_attributes.csv')
        characters_temporal_df = pd.read_csv(story_dir_with_prefix + '.characters_temporal_events.csv')

        return cls(story_id, characters_df, characters_temporal_df, pipeline_dir)

    def 