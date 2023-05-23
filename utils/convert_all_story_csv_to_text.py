# Standard Library
import os

# Local
from src.utils import convert_story_csv_to_txt_file

data_dir = '/Users/pti/challenges/4761_fair_fairytale/fair-fairytale-nlp/data/FairytaleQA/split_by_origin'

fairytale_csvs = []

for path, dirs, files in os.walk(data_dir):
    for csv_file in files:
        if csv_file[-9:] == 'story.csv':
            txt_file = csv_file[:-10] + '.txt'
            convert_story_csv_to_txt_file(os.path.join(path, csv_file), os.path.join(path, txt_file))
            
