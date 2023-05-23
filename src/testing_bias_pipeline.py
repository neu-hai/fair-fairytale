# Standard Library
import json
import os

# Third Party
import pandas as pd

# Local
from src.action_args_pipeline import ActionArgsPipeline
from src.bias_pipeline_batch import BiasPipelineBatch
from src.characters_attributes_pipeline import CharacterAttributesPipeline
from src.characters_sentences_pipeline import CharacterSentencePipeline
from src.srl_pipeline import SRLPipeline, load_AllenNLP_model
from src.supersense_pipeline import SupersensePipeline
from src.temporal_events_pipeline import TemporalEventsPipeline
from src.utils import read_story_txt_file

story_files = ['ali-baba-and-forty-thieves.txt',
               'old-dschang.txt',
               'bamboo-cutter-moon-child.txt',
               'cinderella-or-the-little-glass-slipper.txt',
               'leelinau-the-lost-daughter.txt',
               'the-dragon-princess.txt']

story_id = 'bamboo-cutter-moon-child'
story_dir = '/home/ptoroisaza/fair-fairytale-nlp/data/testing_subset/story_txts/'
#story_path = story_dir + story_id + '.txt'
story_path = os.path.join(story_dir, story_id + '.txt')
pipeline_dir = '/home/ptoroisaza/fair-fairytale-nlp/data/pipeline/'
# pipeline_dir = '/Users/pti/challenges/4761_fair_fairytale/fair-fairytale-nlp/data/pipeline/'

bias_pipeline_batch = BiasPipelineBatch(story_dir, pipeline_dir)
