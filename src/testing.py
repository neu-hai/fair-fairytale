# Standard Library
import json

# Third Party
import pandas as pd

# Local
from src.action_args_pipeline import ActionArgsPipeline
from src.characters_attributes_pipeline import CharacterAttributesPipeline
from src.characters_sentences_pipeline import CharacterSentencePipeline
from src.srl_pipeline import SRLPipeline, load_AllenNLP_model
from src.supersense_pipeline import SupersensePipeline
from src.temporal_events_pipeline import TemporalEventsPipeline

story_files = ['ali-baba-and-forty-thieves.txt',
               'old-dschang.txt',
               'bamboo-cutter-moon-child.txt',
               'cinderella-or-the-little-glass-slipper.txt',
               'leelinau-the-lost-daughter.txt',
               'the-dragon-princess.txt']

story_id = 'bamboo-cutter-moon-child'
pipeline_dir = '/home/ptoroisaza/fair-fairytale-nlp/data/pipeline/'
# pipeline_dir = '/Users/pti/challenges/4761_fair_fairytale/fair-fairytale-nlp/data/pipeline/'

# CharacterSentencePipeline
# character_sentences_pipeline = CharacterSentencePipeline.from_files(story_id, pipeline_dir)

# SRLPipeline
allennlp_model = load_AllenNLP_model()
srl_pipeline = SRLPipeline.from_files(story_id, allennlp_model, pipeline_dir)
print(len(srl_pipeline.srl))

# ActionArgsPipeline
# actions_args_pipeline = ActionArgsPipeline.from_files(story_id, pipeline_dir)

# SupersensePipeline
# supersense_pipeline = SupersensePipeline.from_files(story_id, pipeline_dir)

# CharacterAttributesPipeline
# pipeline_dir = '/Users/pti/challenges/4761_fair_fairytale/fair-fairytale-nlp/data/pipeline/'
# characters_attributes_pipeline = CharacterAttributesPipeline.from_files(story_id, pipeline_dir, ['gender', 'importance'])

# print(characters_attributes_pipeline.story_id)
# print(characters_attributes_pipeline.attributes)
# print(len(characters_attributes_pipeline.sentences_df))
# print(len(characters_attributes_pipeline.sentences_characters_df))
# print(len(characters_attributes_pipeline.characters_df))

# Temporal Events Pipeline
# temporal_events_pipeline = TemporalEventsPipeline.from_files(story_id, pipeline_dir, run_econet = False)




# Saving for Unit Tests

# # print(len(actions_args_pipeline.story_text))
# # print(actions_args_pipeline.sentences_df.head())
# # print(actions_args_pipeline.characters_df.head())
# # print(len(actions_args_pipeline.srl))
# # print(len(actions_args_pipeline.sentences))

# # pos_tags_sentences = actions_args_pipeline.get_pos_tags()
# # print(pos_tags_sentences)
# # print(len(pos_tags_sentences))

# all_sentences_args = actions_args_pipeline.get_all_sentences_args()
# # print(len(all_sentences_args))

# actions_df = actions_args_pipeline.get_coref_ids_in_actions_df(pd.DataFrame(all_sentences_args))
# print(len(actions_df))

# actions_args_pipeline.actions_df = actions_args_pipeline.subset_verbs_with_corefs_in_actions_df(actions_df)
# print(len(actions_args_pipeline.actions_df))

# actions_args_pipeline.set_arg_type_in_actions_df()
# print(actions_args_pipeline.actions_df['arg_type'])

# actions_args_pipeline.convert_long_to_wide_actions_df()
# print(actions_args_pipeline.actions_df.head())

# actions_args_pipeline.clean_actions_df()
# actions_args_pipeline.set_text_offsets_actions_df()
# print(actions_args_pipeline.actions_df.head())

# actions_args_pipeline.save_actions_df_to_csv(actions_args_dir, story_id)