# Standard Library
from collections import defaultdict
import json
from os import listdir, path
import os

# Third Party
import pandas as pd
from tqdm import tqdm

# Local
from src.utils import read_story_txt_file
from newECON.code.main import TemporalEventsModel
from newECON.code.TE_data import get_data
from newECON.code.TE_inference_lib import TE_infer

class TemporalEventsPipeline:
    def __init__(self, story_id: str, pipeline_dir, temporal_events_df: pd.DataFrame, srl: dict, major_event=False):
        self.story_id = story_id
        self.pipeline_dir = pipeline_dir
        self.temporal_events_df = temporal_events_df
        self.srl = srl 
        self.major_event = major_event
        self.rankings_df = self.get_temporal_ranking()

        self.save_rankings_df_to_csv()

        if self.major_event:
            self.major_events_ranking = self.get_temporal_ranking(major_event=True)
            rankings_file = path.join(self.pipeline_dir, self.story_id, self.story_id + '.major_events_temporal_ranks.csv')
            self.major_events_ranking.to_csv(rankings_file, index = False)
    
    @classmethod
    def from_files(cls, story_id, pipeline_dir, run_econet = True, TE_model = None, major_event=False):
        if run_econet == True:
            story_dir_with_prefix = os.path.join(pipeline_dir,story_id, story_id)
            srl = cls.get_srl(story_dir_with_prefix + '.srl.json')
            cls.run_econet(story_id, pipeline_dir, TE_model, major_event=major_event)
            temporal_events_df = pd.read_csv(os.path.join(pipeline_dir, story_id, 'TE_output.csv'))
        else:
            story_dir_with_prefix = pipeline_dir + story_id + '/' + story_id
            temporal_events_df = pd.read_csv(os.path.join(pipeline_dir, story_id, 'TE_output.csv'))
            srl = cls.get_srl(story_dir_with_prefix + '.srl.json')

        return cls(story_id, pipeline_dir, temporal_events_df, srl,major_event=major_event)

    @classmethod
    def get_srl(self, srl_path: str):
        with open(srl_path, 'r') as json_file:
            srl = json.load(json_file)
        return srl

    @classmethod
    def run_econet(self, story_id, pipeline_dir, TE_model,major_event=False):
            if TE_model is None:
                TE_model = TemporalEventsModel()

            story_dir = pipeline_dir + story_id + '/'
            print(story_dir)

            TE_model.check_core_components(story_dir, [story_id])
            TE_model.get_TE_data(pipeline_dir, story_id, get_srl = False, major_event=major_event)

            story_dir = [story_dir]
            model_dir = 'newECON/output/transfer_matres_roberta-large_batch_2_lr_5e-6_epochs_10_seed_23_1.0/'
            model_dir = os.path.abspath(model_dir)
            TE_model.run_TE_inference(story_dir, model_dir)
            print('Saving {} TE_output CSV to: {}'.format(story_id, pipeline_dir + story_id + '/TE_output.csv'))

    def check_srl_te_match(self):
        mis_matches = []
        for s_id in range(len(self.srl)):
            verb_ids = self.temporal_events_df.loc[self.temporal_events_df['s_id'] == s_id, 'e_id'].unique()
            for verb_id in verb_ids:
                srl_verb = self.srl[s_id]['verbs'][verb_id]['verb']
                te_verb = self.temporal_events_df[(self.temporal_events_df['s_id'] == s_id) & (self.temporal_events_df['e_id'] == verb_id)]['event1'].values[0]
                if srl_verb != te_verb:
                    print(s_id, verb_id, srl_verb, te_verb)
                    mis_matches.append((s_id, verb_id))
        if len(mis_matches) > 0:
            print("{} SRL and TE verb ids do not match. Please make sure you have run Econnet with the correct SRL file.")
            return False
        else:
            return True

    def get_temporal_ranking(self, major_event=False):
        if major_event:
            te_df = pd.read_csv(path.join(self.pipeline_dir, self.story_id, 'major_TE_output.csv'))
        else:
            te_df = self.temporal_events_df

        first_row_edge = tuple(te_df.iloc[0, [5, 7]].to_list())
        ranked_list = [first_row_edge]

        for i, row in te_df.iterrows():
            te_rel = row['te_rel']
            edge1 = (row['s_id'], row['e_id'])
            edge2 = (row['s_id2'], row['e_id2'])

            warning = "{} not in ranked_list. All events should appear in columns (s_id, e_id) before appearing in (s_id2, e_id2).".format(edge1)
            
            if te_rel == 'BEFORE':
                if edge1 in ranked_list:
                    edge1_idx = ranked_list.index(edge1)
                    ranked_list.append(edge2)
                else:
                    print(warning)
            elif te_rel == 'AFTER':
                if edge1 in ranked_list:
                    edge1_idx = ranked_list.index(edge1)
                    ranked_list.insert(edge1_idx, edge2)
                else:
                    print(warning)

        rankings_df = pd.DataFrame(ranked_list).rename(columns = {0: 'sentence_id', 1: 'event_id'})
        rankings_df['temporal_rank'] = rankings_df.index
        
        return rankings_df

    def save_rankings_df_to_csv(self):
        rankings_file = path.join(self.pipeline_dir, self.story_id, self.story_id + '.events_temporal_ranks.csv')
        self.rankings_df.to_csv(rankings_file, index = False)
        print('Saving {} events_temporal_ranks CSV to: {}'.format(self.story_id, rankings_file))

def load_TE_model():
    return TemporalEventsModel()
