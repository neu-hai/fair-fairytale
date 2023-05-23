# Standard Library
import json
import os

# Third Party
import pandas as pd

# Local
from src.action_args_pipeline import ActionArgsPipeline
from src.bias_pipeline import BiasPipeline, BiasPipelineBatch
from src.characters_attributes_pipeline import CharacterAttributesPipeline
from src.characters_sentences_pipeline import CharacterSentencePipeline
from src.srl_pipeline import SRLPipeline, load_AllenNLP_model
from src.supersense_pipeline import SupersensePipeline
from src.temporal_events_pipeline import TemporalEventsPipeline
from src.utils import read_story_txt_file

story_dir = './data/FairytaleQA/story_txts/'
pipeline_dir = './data/pipeline/'

stories = ['bamboo-cutter-moon-child']

#bias_pipeline = BiasPipeline(stories[0], story_dir + '/' + stories[0] + '.txt', pipeline_dir, ['temporal_events'])
path_dir = os.path.join(story_dir, stories[0] + '.txt')
bias_pipeline = BiasPipeline(stories[0], path_dir, pipeline_dir, ['temporal_events'] )
# bias_pipeline_batch = BiasPipelineBatch(story_dir, pipeline_dir, stories, 'full')