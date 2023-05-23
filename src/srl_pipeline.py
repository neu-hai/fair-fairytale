# Standard Library
import json
import os
import re
import torch
# Third Party
from allennlp_models import pretrained
import nltk
import pandas as pd

import logging
logging.getLogger("allennlp").setLevel(logging.WARN)

# Local
from src.sentences import split_sentences
from src.utils import read_story_txt_file

class SRLPipeline:
    def __init__(self, story_id: str, sentences: str, AllenNLP_model, pipeline_dir: str):
        self.story_id = story_id 
        self.pipeline_dir = pipeline_dir

        self.story_json = self.get_story_json(sentences)

        try:
            self.srl = AllenNLP_model.predict_batch_json(self.story_json)
        except:
            print("Failed to process story: {}".format(self.story_id))

        self.save_SRL_json()

    @classmethod
    def from_files(cls, story_id: str, AllenNLP_model, pipeline_dir: str):
        story_dir_with_prefix = os.path.join(pipeline_dir, story_id, story_id)
        sentences_path = story_dir_with_prefix + '.sentences.csv'

        sentences = pd.read_csv(sentences_path)['text'].tolist()

        return cls(story_id, sentences, AllenNLP_model, pipeline_dir)

    def get_story_json(self, sentences):
        story_json = []
        for sentence in sentences:
            sentence_json = {}
            sentence_json['sentence'] = sentence
            story_json.append(sentence_json)
        return story_json

    def save_SRL_json(self):
        story_dir_with_prefix = os.path.join(self.pipeline_dir,self.story_id, self.story_id)
        srl_file = story_dir_with_prefix + '.srl.json'
        with open(srl_file, 'w') as json_file:
            json.dump(self.srl, json_file, indent = 4)
        print('Saving {} SRL JSON to: {}'.format(self.story_id, srl_file))

def load_AllenNLP_model():
    if torch.cuda.is_available():
        cuda_device = 0
    else:
        #using cpu
        cuda_device = -1
    return pretrained.load_predictor("structured-prediction-srl", cuda_device=cuda_device)


