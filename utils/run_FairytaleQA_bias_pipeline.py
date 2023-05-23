# Standard Library
import json
import os
import glob
# Third Party
import pandas as pd

# Local
from src.bias_pipeline import BiasPipeline, BiasPipelineBatch

def run_bias():
    story_dir = './data/FairytaleQA/story_txts/'
    pipeline_dir = './data/pipeline/'

    # Test first with a couple of stories
    story_ids = ['bamboo-cutter-moon-child', 'ali-baba-and-forty-thieves']
    story_ids = glob.glob(story_dir+"*.txt")
    story_ids = [os.path.split(x)[1].split('.txt')[0] for x in story_ids]

    bias_pipeline_batch = BiasPipelineBatch(story_dir, pipeline_dir, story_ids, 'full', major_event = True)
run_bias()
