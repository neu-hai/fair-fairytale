import json
from collections import Counter

# with open('../data/stories_all.json') as infile:
#     data = json.load(infile)

# with open('../data/dev_event_sequences_all.json') as infile:
#     dev = json.load(infile)
# data = train + dev

def compute_ngrams(toks, counter, n=1):
    for ngram in [' '.join(toks[t:t+n]) for t in range(len(toks) - n + 1)]:
        counter[str(n)][ngram] += 1

def get_template_arguments(template, relaxed=False):
    if relaxed:
        num_arg_toks = sum([(x.split('-')[-1][:3] == 'ARG' and x.split('-')[-1][-1].isdigit()) or 'ARGM' in x
                            for x in template[3]])
    else:
        num_arg_toks = sum([(x.split('-')[-1][:3] == 'ARG' and x.split('-')[-1][-1].isdigit())
                            for x in template[3]])
    return num_arg_toks

# TODO: Find the most dominant verb in the sentence!
def get_medium_template(templates, relaxed=False):
    temps = sorted([(get_template_arguments(template, relaxed), i) for i, template in enumerate(templates)])
    idx = (len(temps) - 1) // 2
    return templates[temps[idx][1]]

def get_min_template(templates, relaxed=False):
    temps = sorted([(get_template_arguments(template, relaxed), i) for i, template in enumerate(templates)])
    return templates[temps[0][1]]

def get_arg_num(template, relaxed):
    if relaxed:
        num = [x.split('-')[-1] for x in template[3]
               if (x.split('-')[-1][:3] == 'ARG' and x.split('-')[-1][-1].isdigit()) or 'ARGM' in x]
    else:
        num = [x.split('-')[-1] for x in template[3]
               if x.split('-')[-1][:3] == 'ARG' and x.split('-')[-1][-1].isdigit()]
    return len(set(num))

def get_template_with_args(templates, relaxed=False):
    return [t for t in templates if get_arg_num(t, relaxed) >= 2]

def parse_arguments(desc):

    ans = []
    arg = ''
    to_add = False
    for c in desc:
        if c == '[':
            to_add = True
        elif c == ']':
            if arg[:3] == 'ARG' and arg[3].isdigit():
                ans.append((int(arg[3]), arg.split(': ')[1].lower()))
            # if arg[:3] == 'ARG':
            #     ans.append((arg.split(': ')[0], arg.split(': ')[1].lower()))
            arg = ''
            to_add = False
        elif to_add:
            arg += c

    # return at most 2 arguments
    args = [x for i, x in sorted(ans)][:2]
    if len(args) < 2:
        args += [''] * (2 - len(args))

    return args

def make_samples(story_id, story):
    all_tokens = [sent[-1] for sent in story]
    passage = [x for tokens in all_tokens for x in tokens]

    passages = []
    for i in range(len(story)-1):
        sent1 = story[i]
        sent2 = story[i+1]

        # deal with no event case
        if sent1[1] == -1 or sent2[1] == -1:
            passages.append({
                'left': -1,
                'left_event': "",
                'left_arg1': "",
                'left_arg2': "",
                'right': -1,
                'right_event': "",
                'right_arg1': "",
                'right_arg2': "",
                'passage': passage,
                'passage_id': i,
                'story_id': story_id
            })
            continue

        args1 = parse_arguments(sent1[3])
        args2 = parse_arguments(sent2[3])

        assert len(args1) == len(args2) == 2

        sent_lidx, tok_lidx = sent1[0], sent1[1]
        sent_ridx, tok_ridx = sent2[0], sent2[1]

        loffset = sum([len(sent) for i, sent in enumerate(all_tokens) if i + 1 < sent_lidx])
        roffset = sum([len(sent) for i, sent in enumerate(all_tokens) if i + 1 < sent_ridx])

        assert passage[loffset + tok_lidx] == sent1[-1][tok_lidx]
        assert passage[roffset + tok_ridx] == sent2[-1][tok_ridx]

        passages.append({
                        'left': loffset + tok_lidx,
                        'left_event': sent1[-1][tok_lidx],
                        'left_arg1': args1[0],
                        'left_arg2': args1[1],
                        'right': roffset + tok_ridx,
                        'right_event': sent2[-1][tok_ridx],
                        'right_arg1': args2[0],
                        'right_arg2': args2[1],
                        'passage': passage,
                        'passage_id': i,
                        'story_id': story_id
                        })

    return passages

files = ['_spring2016', '_winter2017']

#files = ['_test_missing']
all_data = []
c1, c2 = Counter(), Counter()
event_counter, trigger_counter = Counter(), Counter()

N = 3
a1_ngrams = {str(n+1): Counter() for n in range(N)}
a2_ngrams = {str(n+1): Counter() for n in range(N)}
template_choices = [0]*3
for f in files:
    with open('../output/stories%s.json' % f) as infile:
        data = json.load(infile)

    for k, story in data.items():
        sents = []
        for sent in story:
            if sent[0][1] == -1:
                sents.append(sent[0])
                continue

            templates = get_template_with_args(sent)
            if templates:
                template = get_min_template(templates)
                template_choices[0] += 1
            else:
                templates = get_template_with_args(sent, relaxed=True)

                if templates:
                    template = get_min_template(templates, relaxed=True)
                    template_choices[1] += 1
                else:
                    template = get_medium_template(sent, relaxed=True)
                    template_choices[2] += 1

            sents.append(template)

            # trigger = template[2][template[1]]
            # trigger_counter[trigger] += 1
            # args = parse_arguments(template[4])
            #
            # l1 = 0 if args[0] == '' else len(args[0].split(' '))
            # l2 = 0 if args[1] == '' else len(args[1].split(' '))
            # c1[l1] += 1
            # c2[l2] += 1
            #
            # compute_ngrams(args[0].split(' '), a1_ngrams, n=1)
            # compute_ngrams(args[0].split(' '), a1_ngrams, n=2)
            # compute_ngrams(args[0].split(' '), a1_ngrams, n=3)
            # compute_ngrams(args[1].split(' '), a2_ngrams, n=1)
            # compute_ngrams(args[1].split(' '), a2_ngrams, n=2)
            # compute_ngrams(args[1].split(' '), a2_ngrams, n=3)

            # event_counter["%s %s %s" % (args[0], trigger, args[1])] += 1
        all_data.extend(make_samples(k, sents))

with open('../output/stories_all_complete.json', 'w') as outfile:
    json.dump(all_data, outfile, indent=4)

# with open('../output/stories_all_complete.json', 'w') as outfile:
#     json.dump(all_data, outfile, indent=4)
#
# assert sum(c1.values()) == sum(c2.values())
# total = sum(c1.values())
# print(sorted([(k, 100.0 * v / total) for k, v in c1.items()]))
# print(sorted([(k, 100.0 * v / total) for k, v in c2.items()]))
#
# print("=== Trigger ===")
# print(trigger_counter.most_common(50))
# print("=== Argument 1 ===")
# for n in range(N):
#     print(n+1)
#     print(a1_ngrams[str(n+1)].most_common(50))
# print("=== Argument 2 ===")
# for n in range(N):
#     print(n+1)
#     print(a2_ngrams[str(n+1)].most_common(50))
#
# print("=== Events ===")
# print(event_counter.most_common(50))
#
# print(template_choices)