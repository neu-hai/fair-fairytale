import os
import re
import csv

from typing import List

import pandas as pd

from src.sentences import split_sentences 
from src.utils import read_story_txt_file

class TemporalEventsArgsPipeline:
	def __init__(self, story_id: str, sentences_df: pd.DataFrame, action_args_df: pd.DataFrame, temporal_events_df: pd.DataFrame):
		pass