import pandas as pd
from scipy.stats import mode
from collections import Counter
from utils import *
import random
import json

def merge_dfs(df1, df2, df3, has_gold=False):
    df = pd.merge(pd.merge(df1, df2, on='PassageKey'), df3, on='PassageKey')
    if has_gold:
        df.columns = ['PassageKey', 'Prediction_x', 'Gold_x', 'Prediction_y', 'Gold_y', 'Prediction_z', 'Gold_z']
    else:
        df.columns = ['PassageKey', 'Prediction_x', 'Prediction_y', 'Prediction_z']
    return df

def majority(row):
    preds = mode([row['Prediction_x'], row['Prediction_y'], row['Prediction_z']])
    return preds[0][0] if preds[1][0] >= 2 else 'VAGUE'

def strict_count(row):
    preds = mode([row['Prediction_x'], row['Prediction_y'], row['Prediction_z']])
    return preds[0][0] if preds[1][0] == 3 else 'VAGUE'

def single_edge_distribution(df):

    df['majority'] = df.apply(lambda row: majority(row), axis=1)
    df['strict'] = df.apply(lambda row: strict_count(row), axis=1)

    print("Single Edge Distribution by majority vote.")
    print(Counter(df['majority'].tolist()))

    print("Single Edge Distribution by consensus.")
    print(Counter(df['strict'].tolist()))
    return df


def get_story_ids(df):
    df['story_id'] = df.apply(lambda row: row['PassageKey'][:-2], axis=1)
    return df

def categorize(x):
    ordering = list(set(x.tolist()))
    if len(ordering) == 1:
        return ordering[0]
    else:
        return 'VAGUE'

def print_sequence(events, relations):
    idx = 0
    assert len(events) == len(relations) + 1
    output = ""
    while idx < len(relations):
        output += "%s -- %s -- " % (events[idx], relations[idx])
        idx += 1
    output += events[idx]
    print(output)

storyid_title_map = {}
storyid_story_map = {}
count = 0
for suffix in ['spring2016', 'winter2017']:
    data = pd.read_csv("../data/ROCStories_%s.csv" % suffix)
    for r, row in data.iterrows():
        storyid_title_map[row['storyid']] = row['storytitle']
        story = []
        for s in range(1, 6):
            temp = row['sentence%s' % s]
            while temp[-1] in ['"', '\\', '/', ',', ' ']:
                temp = temp[:-1]
            if temp[-1] not in ['.', '?', '!']:
                count += 1
                temp += '.'

            story.append(temp)

        storyid_story_map[row['storyid']] = ' '.join(story)

print(len(storyid_title_map), len(storyid_story_map))

data_dir = "../output/"

suffix = "all_complete_test_missing"
df1 = pd.read_csv('%spred_%s_%s.csv' % (data_dir, 5, suffix))
print(Counter(df1['Prediction'].tolist()))

df2 = pd.read_csv('%spred_%s_%s.csv' % (data_dir, 7, suffix))
print(Counter(df2['Prediction'].tolist()))

df3 = pd.read_csv('%spred_%s_%s.csv' % (data_dir, 23, suffix))
print(Counter(df3['Prediction'].tolist()))

print(df1.shape, df2.shape, df3.shape)

df = merge_dfs(df1, df2, df3, has_gold=False)
print(df.shape)
df = get_story_ids(df)
df = single_edge_distribution(df)
print(df.head())

with open("../output/stories_%s.json" % suffix) as infile:
    data = json.load(infile)

lookup = {}
for ex in data:
    key = "%s-%s" % (ex['story_id'], ex['passage_id'])
    lookup[key] = [(ex['left_event'].lower(), ex['left_arg1'], ex['left_arg2']),
                   (ex['right_event'].lower(), ex['right_arg1'], ex['right_arg2'])]

samples = []
events, relations = [], []
prev_story = ''
for r, row in df.iterrows():
    if prev_story and row['story_id'] != prev_story:
        samples.append({'story_id': prev_story, 'title': storyid_title_map[prev_story],
                        'events': events, 'relations': relations, 'story': storyid_story_map[prev_story]})
        #print_sequence(events, relations)
        events, relations = [], []

    key = row['PassageKey']
    relations.append(row['strict'])
    if key.split('-')[-1] == '0':
        events.extend(lookup[key])
    else:
        events.append(lookup[key][1])

    prev_story = row['story_id']

samples.append({'story_id': prev_story, 'title': storyid_title_map[prev_story],
                'events': events, 'relations': relations, 'story': storyid_story_map[prev_story]})
print(len(samples))

with open("../data/test_story_generation_all_complete.json") as infile:
    test_data = json.load(infile)
    test_data += samples
    print(len(test_data))

with open("../data/test_story_generation_all_complete_fix_missing.json", 'w') as outfile:
    json.dump(test_data, outfile)


with open("../data/dev_story_ids.json") as infile:
    dev_ids = json.load(infile)

with open("../data/test_story_ids.json") as infile:
    test_ids = json.load(infile)


print(len(dev_ids), len(test_ids))

train, dev, test = [], [], []
for sample in samples:
    if sample['story_id'] in dev_ids:
        dev.append(sample)
    elif sample['story_id'] in test_ids:
        test.append(sample)
    else:
        train.append(sample)
print(len(train), len(dev), len(test))

# random.Random(7).shuffle(samples)
# split = int(len(samples) * 0.9)
#save_name = "event_sequences_%s" % suffix
save_name = "story_generation_%s" % suffix
with open("../data/train_%s.json" % save_name, 'w') as outfile:
    json.dump(train, outfile, indent=2)

with open("../data/dev_%s.json" % save_name, 'w') as outfile:
    json.dump(dev, outfile, indent=2)

with open("../data/test_%s.json" % save_name, 'w') as outfile:
    json.dump(test, outfile, indent=2)

# seq_strict = df.groupby(['story_id'])['strict'].agg([categorize])
# print(Counter(seq_strict['categorize'].tolist()))
#
# seq_major = df.groupby(['story_id'])['majority'].agg([categorize])
# print(Counter(seq_major['categorize'].tolist()))

# golds, preds_major, preds_strict = [], [], []
# for_analysis = []
# for r, row in df.iterrows():
#     golds.append(row['Gold_x'])
#     preds_major.append(row['majority'])
#     preds_strict.append(row['strict'])
#
#     if row['strict'] == 'AFTER' and row['Gold_x'] == 'VAGUE':
#         for_analysis.append(row['PassageKey'])
#
# random.Random(7).shuffle(for_analysis)
# print(for_analysis[:20])

# report = ClassificationReport('caters-major', golds, preds_major)
# print(report)
# report = ClassificationReport('caters-strict', golds, preds_strict)
# print(report)

# df1a = pd.read_csv('%spred_%s_2016.csv' % (data_dir, 5))
# df1b = pd.read_csv('%spred_%s_2017.csv' % (data_dir, 5))
# df1 = pd.concat([df1a, df1b])
# print(df1a.shape, df1b.shape, df1.shape)
# print(Counter(df1['Prediction'].tolist()))
#
# df2a = pd.read_csv('%spred_%s_2016.csv' % (data_dir, 7))
# df2b = pd.read_csv('%spred_%s_2017.csv' % (data_dir, 7))
# df2 = pd.concat([df2a, df2b])
# print(Counter(df2['Prediction'].tolist()))
#
# df3a = pd.read_csv('%spred_%s_2016.csv' % (data_dir, 23))
# df3b = pd.read_csv('%spred_%s_2017.csv' % (data_dir, 23))
# df3 = pd.concat([df3a, df3b])
# print(Counter(df3['Prediction'].tolist()))
#
# print(df1.shape, df2.shape, df3.shape)
#
# df = merge_dfs(df1, df2, df3)
# print(df.shape)
# df = get_story_ids(df)
# df = single_edge_distribution(df)
# print(df.head())
#
# seq_strict = df.groupby(['story_id'])['strict'].agg([categorize])
# print(Counter(seq_strict['categorize'].tolist()))
#
# seq_major = df.groupby(['story_id'])['majority'].agg([categorize])
# print(Counter(seq_major['categorize'].tolist()))