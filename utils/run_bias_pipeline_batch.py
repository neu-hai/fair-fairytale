# Standard Library
from argparse import ArgumentParser
import json
import os
import sys

# Third Party
import pandas as pd

# Local
from src.bias_pipeline import BiasPipelineBatch

def main(argv):
    parser = ArgumentParser(description="Temporal event bias pipeline for agents and patients.")
    parser.add_argument('--story-dir',
                        '-sd',
                        required=True,
                        help="Directory with story texts.")
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
    parser.set_defaults(major_events = False)
    parser.set_defaults(pronoun_only_entities = False)
    parser.set_defaults(keep_all_verbs = False)

    args = parser.parse_args(argv)

    # Get story_ids from story_dir
    story_files = os.listdir(args.story_dir)
    story_ids = []
    for story_file in story_files:
        if '.txt' in story_file:
            story_id = story_file[:-4]
            story_ids.append(story_id)

    pipelines = args.pipelines.split(',')

    bp = BiasPipelineBatch(story_dir = args.story_dir, pipeline_dir = args.pipeline_dir, story_ids = story_ids,
                           pipelines = pipelines, major_event = args.major_events,
                           pronoun_only_entities = args.pronoun_only_entities, keep_all_verbs = args.keep_all_verbs)

if __name__ == "__main__":
    main(sys.argv[1:])
