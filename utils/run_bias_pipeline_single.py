# Standard Library
from argparse import ArgumentParser
import json
import sys

# Third Party
import pandas as pd

# Local
from src.bias_pipeline import BiasPipeline

def main(argv):
    parser = ArgumentParser(description="Temporal event bias pipeline for agents and patients.")
    parser.add_argument('--story-infile',
                        '-i',
                        required=True,
                        help="Path to the story text.")
    parser.add_argument('--story-id',
                        '-si',
                        required=False,
                        help="Story id used for saving files. If not provided, the story id will be the file name of the story infile.")
    parser.add_argument('--pipeline-dir',
                        '-pd',
                        required=True,
                        help="Directory for the pipeline output.")
    parser.add_argument('--pipelines',
                        '-p',
                        default='full',
                        help="List of pipelines to run. Default is full pipeline. Options: full, booknlp, characters_sentences, attributes, srl, actions_args, supersense, temporal_events, temporal_events_characters, statistics")
    parser.add_argument('--major-events',
                        '-me',
                        dest='major_events',
                        action='store_true',
                        help="Flag to find major events.")
    parser.add_argument('--pronoun-only-entities',
                        '-poe',
                        dest='pronoun_only_entities',
                        action='store_true',
                        help="Flag to keep character entities only referred to with pronouns.")
    parser.add_argument('--keep-all-verbs',
                        '-kav',
                        action='store_true',
                        dest='keep_all_verbs',
                        help="Flag to keep stative verbs.")
    parser.set_defaults(major_events = True)
    parser.set_defaults(pronoun_only_entities = False)
    parser.set_defaults(keep_all_verbs = False)

    args = parser.parse_args(argv)

    # Get story_id from story_infile
    if 'args.story_id' in locals():
        story_id = args.story_id 
    else:
        last_slash_idx = args.story_infile.rindex('/')
        story_id = args.story_infile[last_slash_idx + 1:-4]

    pipelines = args.pipelines.split(',')
    
    pdir = args.pipeline_dir
    if not pdir.endswith('/'):
        pdir += '/'
    bp = BiasPipeline(story_id = story_id, story_path = args.story_infile, pipeline_dir = pdir, 
                      pipelines = pipelines, major_event = args.major_events, 
                      pronoun_only_entities = args.pronoun_only_entities, keep_all_verbs = args.keep_all_verbs)

if __name__ == "__main__":
    main(sys.argv[1:])
