import pandas as pd
from utils import *
# from transformers import *
import transformers
import nltk
from collections import Counter
import json
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from allennlp_models.pretrained import load_predictor
# predictor = load_predictor("structured-prediction-srl-bert", cuda_device=0)
predictor = load_predictor("structured-prediction-srl-bert")
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def get_predicate(sentence):
    preds = predictor.predict(sentence)
    events = []
    try:
        for pred in preds["verbs"]:
            events.append((pred["verb"], pred["tags"].index('B-V'), pred["tags"], pred["description"], preds['words']))
    except:
        logger.info('No Event!')
        logger.info(sentence)
    return events

def get_event_idx(sent, event):
    tokens = nltk.word_tokenize(sent)
    idx = -1
    ### TODO: check all multiple identical event span handled properly
    ### TODO: check error rate
    min_dist = float('inf')
    for i, tok in enumerate(tokens):
        if tok == event[0]:
            dist = abs(i - event[1])
            if dist < min_dist:
                min_dist = dist
                idx = i
    return idx, tokens

def pair_sents(story):
    passages = []
    for i in range(len(story)-1):
        sent1 = story[i]
        sent2 = story[i+1]
        passages.append({
                        'left': sent1[0],
                        'left_event': sent1[1],
                        'right': sent2[0] + len(sent1[2]),
                        'right_event': sent2[1],
                        'passage': sent1[2] + sent2[2],
                        'passage_id': i,
                        'story_id': sent1[3]
                        })
    return passages

if __name__ == "__main__":
    data_dir = '/lfs1/rujunhan/Event-Forecast-NLG/data/'
    file = 'ROCStories_winter2017.csv'

    data = pd.read_csv(data_dir + file)

    samples = {}

    # with open(data_dir + "caters_roc_ids.json") as infile:
    #     caters_stories = json.load(infile)

    for split in ['test']:
        with open("../data/%s_story_ids.json" % split) as infile:
            test_story_ids = json.load(infile)

        with open("../data/%s_story_generation_all_complete.json" % split) as infile:
            test_story_ids_exist = [ex['story_id'] for ex in json.load(infile)]

        story_ids = [i for i in test_story_ids if i not in test_story_ids_exist]
        print(len(test_story_ids), len(test_story_ids_exist), len(story_ids))

    char_len, tok_len = [], []
    counter = 0

    for r, row in data.iterrows():
        if row['storyid'] not in story_ids:
            continue

        story, idx, tokens = [], [], []
        for i in range(1, 6):
            sent = row['sentence%s' % i]
            #char_len.append(len(sent))
            #tok_len.append(len(sent.split(' ')))
            events = get_predicate(sent)
            if not events:
                break

            templates = []
            for event in events:
                idx, tokens = get_event_idx(sent, event)
                if idx == -1:
                    logger.info("NLTK Mismatch!")
                    logger.info(event)
                else:
                    templates.append((i, idx, tokens, event[2], event[3], event[4]))
            if not templates:
                break

            story.append(templates)

        if len(story) == 5:
            samples[row['storyid']] = story


        if (r+1) % 10000 == 0:
            logger.info("Processed %s stories so far!" % (r+1))
            logger.info("Total %s samples" % len(samples))

    # logger.info("Global Max Length is %s" % global_max_len)
    logger.info("Total %s samples" % len(samples))
    with open('../output/stories_spring2016.json', 'w') as outfile:
        json.dump(samples, outfile, indent=2)

    print(np.mean(char_len), max(char_len))
    print(np.mean(tok_len), max(tok_len), min(tok_len))
    print(Counter(tok_len))