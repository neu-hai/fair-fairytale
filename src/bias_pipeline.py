# Standard Library
import json
import os
import re
import csv
from os import path
# Third Party
import pandas as pd
from tqdm import tqdm
import torch
# Local
from src.action_args_pipeline import ActionArgsPipeline
from src.booknlp_pipeline import BookNLPPipeline, load_BookNLP_model
from src.characters_attributes_pipeline import CharacterAttributesPipeline
from src.characters_sentences_pipeline import CharacterSentencePipeline
from src.characters_temporal_events_pipeline import TemporalEventsCharactersPipeline
from src.major_actions_args import get_major_actions
from src.statistics import Statistics
from src.srl_pipeline import SRLPipeline, load_AllenNLP_model
from src.supersense_pipeline import SupersensePipeline
from src.temporal_events_pipeline import TemporalEventsPipeline, load_TE_model
from src.utils import read_story_txt_file, get_evls, get_events_df
import os, os.path
import errno

# Taken from https://stackoverflow.com/a/600612/119527
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def safe_open_w(path):
    ''' Open "path" for writing, creating any parent directories as needed.
    '''
    mkdir_p(os.path.dirname(path))
    return open(path, 'w')


class BiasPipeline:
    def __init__(self, story_id: str, story_path: str, pipeline_dir: str, 
                 pipelines = ['full'], batch = False, 
                 bookNLP_model = None, allennlp_model = None, TE_model = None,
                 pronoun_only_entities = False, keep_all_verbs = False, major_event = False):
        self.story_id = story_id
        self.story_path = story_path
        self.pipeline_dir = pipeline_dir
        self.story_text = read_story_txt_file(story_path)
        self.major_event= major_event
        # Write story text file in pipeline_dir
        with safe_open_w(os.path.join(self.pipeline_dir, story_id, story_id + '.txt')) as f:
            f.write(self.story_text)

        if 'full' in pipelines or 'booknlp' in pipelines:
            try:
                # Run BookNLP+GX
                # print("Running booknlp_pipeline...")
                if batch == False:
                    bookNLP_model = load_BookNLP_model()
                booknlp_pipeline = BookNLPPipeline(story_id, story_path, bookNLP_model, pipeline_dir)

                self.tokens_df = booknlp_pipeline.tokens_df
                self.entities_df = booknlp_pipeline.entities_df
                self.character_meta_df = booknlp_pipeline.character_meta_df
                self.supersense_df = booknlp_pipeline.supersense_df
            except:
                print("Could not run booknlp on: {}".format(story_id))

        # Run Characters Sentences Pipeline
        if 'full' in pipelines or 'characters_sentences' in pipelines:
            try:
                # print("Running characters_sentences_pipeline...")
                if any(pipeline in pipelines for pipeline in ['full', 'booknlp']):
                    characters_sentences_pipeline = CharacterSentencePipeline(story_id, self.entities_df, self.tokens_df, self.story_text, pipeline_dir, pronoun_only_entities)
                else:
                    characters_sentences_pipeline.from_files(story_id, pipeline_dir, pronoun_only_entities)

                # print(len(characters_sentences_pipeline.sentences_df))
                # print(len(characters_sentences_pipeline.sentences_by_character_df))

                self.sentences_df = characters_sentences_pipeline.sentences_df
                self.sentences_by_character_df = characters_sentences_pipeline.sentences_by_character_df
            except:
                print("Could not run character_sentences on: {}".format(story_id))

        # Run Character Attributes Pipeline
        if 'full' in pipelines or 'attributes' in pipelines:
            print("Running characters_attributes_pipeline...")
            try:
                if any(pipeline in pipelines for pipeline in ['full', 'character_sentences']):
                    characters_attributes_pipeline = CharacterAttributesPipeline(story_id, pipeline_dir, 
                                                                                 self.sentences_df, self.sentences_by_character_df, self.character_meta_df, 
                                                                                 ['gender', 'importance'])
                else:
                    characters_attributes_pipeline = CharacterAttributesPipeline.from_files(story_id, pipeline_dir, ['gender', 'importance'])

                # print(len(characters_attributes_pipeline.characters_df))
                # print(characters_attributes_pipeline.characters_df.head())

                self.characters_df = characters_attributes_pipeline.characters_df
            except:
                print("Could not run characters_attributes on: {}".format(story_id))

        # Run AllenNLP SRL
        if 'full' in pipelines or 'srl' in pipelines:
            print("Running srl_pipeline...")
            try:
                with torch.no_grad():
                    if batch == False:
                        allennlp_model = load_AllenNLP_model()

                    if any(pipeline in pipelines for pipeline in ['full', 'characters_sentences']):
                        srl_pipeline = SRLPipeline(story_id, self.sentences_df['text'].tolist(), allennlp_model, pipeline_dir)
                    else:
                        srl_pipeline = SRLPipeline.from_files(story_id, allennlp_model, pipeline_dir)
                    del allennlp_model
                    torch.cuda.empty_cache()
                self.srl = srl_pipeline.srl
            except:
                print("Could not run SRL on: {}".format(story_id))

        # Run Action Args Pipeline
        if 'full' in pipelines or 'actions_args' in pipelines:
            try:
                if any(pipeline in pipelines for pipeline in ['full']):
                    action_args_pipeline = ActionArgsPipeline(story_id, self.story_text, self.sentences_df, self.sentences_by_character_df, self.srl, pipeline_dir)
                else:
                    action_args_pipeline = ActionArgsPipeline.from_files(story_id, pipeline_dir)
                self.actions_args_df = action_args_pipeline.actions_df
            except:
                print("Could not run action_args on: {}".format(story_id))

        # Run Supersense Pipeline
        if 'full' in pipelines or 'supersense' in pipelines:
            try:
                supersense_pipeline = SupersensePipeline.from_files(story_id, pipeline_dir, keep_all_verbs)
                print("Adding supersense categories to {}.actions_args.csv".format(story_id))
            except FileNotFoundError:
                print("Did not add supersense categories to {}.actions_args.csv".format(story_id))

        # Major Events 
        if self.major_event:
            event_coref = pd.read_csv(path.join(pipeline_dir, story_id, story_id + '.actions_args.csv'))
            major_actions_df = get_major_actions(event_coref, self.story_text)
            major_actions_df.to_csv(path.join(pipeline_dir, story_id, 'major_actions_args.csv'), index=None)
            self.major_actions_df = major_actions_df

        # Run EventPlus
        if 'full' in pipelines or 'temporal_events' in pipelines:
            try:
                temporal_events_pipeline = TemporalEventsPipeline.from_files(story_id, pipeline_dir, major_event=major_event)
                self.temporal_events_df = temporal_events_pipeline.temporal_events_df
                # TODO: if action_args exist: assume it does for now; create predicted_major_action file, for annotation and eyeballing.
                if self.major_event:
                    events_df = pd.read_csv(path.join(pipeline_dir, story_id, story_id + '.actions_args.csv'))
                    evls = get_evls(events_df)
                    if os.path.isfile(path.join(pipeline_dir, story_id, 'major_actions_args.csv')):
                        major = pd.read_csv(path.join(pipeline_dir, story_id, 'major_actions_args.csv'))
                        sents = self.sentences_df['text'].to_list()
                        pred_events_df = get_events_df(major, evls, sents)
                        pred_events_df.to_csv(path.join(pipeline_dir, story_id, 'pred_major_actions.csv'),index=None)
                    else:
                        print('error, no major events')
            except:
                print("Could not run temporal_events on: {}".format(story_id))

        # Run Temporal Events Characters Pipeline 
        if 'full' in pipelines or 'temporal_events_characters' in pipelines:
            try:
                temporal_events_characters_pipeline = TemporalEventsCharactersPipeline.from_files(story_id, pipeline_dir, major_event)
                self.characters_temporal_df = temporal_events_characters_pipeline.characters_temporal_df
                self.subj_dobj_link_df = temporal_events_characters_pipeline.subj_dobj_link_df
                if major_event:
                    self.major_characters_temporal_df = temporal_events_characters_pipeline.major_characters_temporal_df
                else:
                    self.major_characters_temporal_df = None
            except:
                print("Could not run temporal_events_characters on: {}".format(story_id))
           
        # Run Statistics
        if 'full' in pipelines or 'statistics' in pipelines:
            try:
                if 'full' in pipelines:
                    self.statistics = Statistics(story_id, self.characters_df, self.characters_temporal_df, self.sentences_df, pipeline_dir, self.major_characters_temporal_df)
                else:
                    self.statistics = Statistics.from_files(story_id, pipeline_dir, major_event)
            except:
                print("Could not run statistics on: {}".format(story_id))

class BiasPipelineBatch:
    def __init__(self, input_dir: str, pipeline_dir: str, story_ids = [], 
                 pipelines = ['full'], pronoun_only_entities = False, keep_all_verbs = False,
                 major_event=False):
        """

        :param input_dir:
        :param pipeline_dir:
        :param story_ids:
        :param pipelines:
        :param major_event: Flag for whether to run extraction of major event-chains.
        """
                
        self.pipeline_dir = pipeline_dir
        self.story_ids = self.get_story_ids(input_dir, story_ids)
        print("Running pipeline on {} stories...".format(len(self.story_ids)))

        # Load models first
        if any(pipeline in pipelines for pipeline in ['full', 'booknlp']):
            print("Loading BookNLP model...")
            bookNLP_model = load_BookNLP_model()
            print("Done loading...")
        else:
            bookNLP_model = None 
        if any(pipeline in pipelines for pipeline in ['full', 'srl']):
            print("Loading AllenNLP SRL model...")
            allennlp_model = load_AllenNLP_model()
            print("Done loading...")
        else:
            allennlp_model = None
        if any(pipeline in pipelines for pipeline in ['full', 'temporal_events']):
            print("Loading temporal events model...")
            TE_model = load_TE_model()
            print("Done loading...")
        else:
            TE_model = None

        bias_pipelines = []

        for story_id in tqdm(self.story_ids):
            print("Running {}...".format(story_id))
            story_path = input_dir + story_id + '.txt'
            bias_pipelines.append(BiasPipeline(story_id, story_path, pipeline_dir, pipelines, True, bookNLP_model, allennlp_model, TE_model, pronoun_only_entities, keep_all_verbs, major_event))

    def get_story_ids(self, input_dir, story_ids):
        input_files = os.listdir(input_dir)
        if len(story_ids) == 0:
            story_ids = [story_txt_file[:-4] for story_txt_file in input_files if story_txt_file[-4:] == '.txt']
        return story_ids