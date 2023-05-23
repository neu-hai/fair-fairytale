# Standard Library
import json
import os

# Third Party
import pandas as pd

# Local
from src.booknlp_pipeline import BookNLPPipeline, load_BookNLP_model

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
output_dir = '/home/ptoroisaza/fair-fairytale-nlp/data/pipeline/'

print(story_path)

bookNLP_model = load_BookNLP_model()

booknlp_pipeline = BookNLPPipeline(story_id, story_path, bookNLP_model, output_dir)