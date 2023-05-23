
import os
import pandas as pd
from spacy.tokens import Doc
import spacy
import re


class Token:

    def __init__(self, paragraph_id, sentence_id, index_within_sentence_idx, token_id, text, pos, fine_pos, lemma, deprel, dephead, ner, startByte):
        self.text=text
        self.paragraph_id=paragraph_id
        self.sentence_id=sentence_id
        self.index_within_sentence_idx=index_within_sentence_idx
        self.token_id=token_id
        self.lemma=lemma
        self.pos=pos
        self.fine_pos=fine_pos
        self.deprel=deprel
        self.dephead=dephead
        self.ner=ner
        self.startByte=startByte
        self.endByte=startByte+len(text)
        self.inQuote=False
        self.event="O"

    def __str__(self):
        return '\t'.join([str(x) for x in [self.paragraph_id, self.sentence_id, self.index_within_sentence_idx, self.token_id, self.text, self.lemma, self.startByte, self.endByte, self.pos, self.fine_pos, self.deprel, self.dephead, self.event]])

    @classmethod
    def convert(self, sents):
        toks=[]
        i=0
        cur=0
        for sidx, sent in enumerate(sents):
            for widx, word in enumerate(sent):
                token=Token(0, sidx,widx,i,word, None, None, None, None, None, None, cur)
                toks.append(token)
                i+=1
                cur+=len(word) + 1
        return toks

    @classmethod
    def deconvert(self, toks):
        sents=[]
        sent=[]
        lastSid=None
        for tok in toks:
            if lastSid is not None and tok.sentence_id != lastSid:
                sents.append(sent)
                sent=[]
            sent.append(tok)
            lastSid=tok.sentence_id

        if len(sent) > 0:
            sents.append(sent)

        # print(sents)
        return sents


class SpacyPipeline:
    def __init__(self, spacy_nlp):
        self.spacy_nlp=spacy_nlp
        self.spacy_nlp.max_length = 10000000


    def filter_ws(self, text):
        text=re.sub(" ", "S", text)
        text=re.sub("[\n\r]", "N", text)
        text=re.sub("\t", "T", text)
        return text


    def tag_pretokenized(self, toks, sents, spaces):

        doc = Doc(self.spacy_nlp.vocab, words=toks, spaces=spaces)
        for idx, token in enumerate(doc):
            token.sent_start=sents[idx]

        for name, proc in self.spacy_nlp.pipeline:
            doc = proc(doc)

        return self.process_doc(doc)

    def tag(self, text):

        doc = self.spacy_nlp(text)
        return self.process_doc(doc)

    def process_doc(self, doc):

        tokens=[]
        skipped_global=0
        paragraph_id=0
        current_whitespace=""
        sentence_id=0
        for sid, sent in enumerate(doc.sents):
            skipped_in_sentence=0
            skips_in_sentence=[]
            curSkips=0
            for w_idx, tok in enumerate(sent):
                if tok.is_space:
                    curSkips+=1
                skips_in_sentence.append(curSkips)

            hasWord=False

            for w_idx, tok in enumerate(sent):

                if tok.is_space:
                    skipped_global+=1
                    skipped_in_sentence+=1
                    current_whitespace+=tok.text
                else:
                    if re.search("\n\n", current_whitespace) is not None:
                        paragraph_id+=1

                    hasWord=True

                    head_in_sentence=tok.head.i-sent.start
                    skips_between_token_and_head=skips_in_sentence[head_in_sentence]-skips_in_sentence[w_idx]
                    token=Token(paragraph_id, sentence_id, w_idx-skipped_in_sentence, tok.i-skipped_global, self.filter_ws(tok.text), tok.pos_, tok.tag_, tok.lemma_, tok.dep_, tok.head.i-skipped_global-skips_between_token_and_head, None, tok.idx)
                    tokens.append(token)
                    current_whitespace=""

            if hasWord:
                sentence_id+=1

        return tokens







spacy_model="en_core_web_sm"
spacy_nlp = spacy.load(spacy_model, disable=["ner"])
tagger=SpacyPipeline(spacy_nlp)
# this is the token_creating tagger by booknlp; it ensures alignment with booknlp results


def append_supersense(dir, story):
    """
    :param dir: the dir underwhich which expect 3 files,
            1. story.txt
            2. story.supersense
            3. story.verb_subj_dobj.srl.csv
    :param story: the name of the story to process
    :return: None;
            it updates the story.verb_subj_dobj.srl.csv file with an additional supersense column
    """

    text_file = os.path.join(dir, story+'.txt')
    super_file = os.path.join(dir, story +'.supersense')
    event_coref_file = os.path.join(dir, story +'.verb_subj_dobj.srl.csv')

    with open(text_file) as f:
        data = f.read()
    tokens = tagger.tag(data)
    # super_df = pd.read_csv(super_file, sep='\t')
    super_df = pd.read_csv(super_file, sep=',')
    event_coref = pd.read_csv(event_coref_file)


    sBytes, eBytes = [], []
    for idx in range(super_df.shape[0]):
        start_token = super_df.iloc[idx]['start_token']
        end_token = super_df.iloc[idx]['end_token']

        token_obj = tokens[start_token]
        end_token_obj = tokens[end_token]
        token_start_byte, token_end_byte = token_obj.startByte, end_token_obj.endByte
        sBytes.append(token_start_byte)
        eBytes.append(token_end_byte)
    super_df['StartByte'] = sBytes
    super_df['EndByte'] = eBytes
    super_df.to_csv(super_file, index=None)

    event_label = []
    for idx in range(event_coref.shape[0]):
        verb = event_coref.iloc[idx]['verb']
        verb_start_byte = event_coref.iloc[idx]['verb_start_byte_text']

        super_line = super_df.loc[super_df['StartByte'] == verb_start_byte]
        if super_line.shape[0]:
            supersense = super_line.iloc[0]['supersense_category']

            assert verb in super_line.iloc[0]['text']

            if ('verb' in supersense) and (supersense != 'verb.stative'):
                event_label.append(1)
            else:
                event_label.append(0)
        else:
            event_label.append(0)
    event_coref['event_label'] = event_label
    event_coref.to_csv(event_coref_file, index=None)



if __name__ == "__main__":
    stories = ['ali-baba-and-forty-thieves', 'old-dschang','cinderella-or-the-little-glass-slipper'
        ,'bamboo-cutter-moon-child','leelinau-the-lost-daughter','the-dragon-princess']
    # stories= ['the-dragon-princess']
    for i in range(6):
        dir = "../may18_data"
        append_supersense(dir, stories[i])

        #
        # text_file = os.path.join("../may18_data", stories[i]+'.txt')
        # super_file = os.path.join("../may18_data", stories[i]+'.supersense')
        # event_coref_file = os.path.join("../may18_data", stories[i]+'.verb_subj_dobj.srl.csv')
        #
        # with open(text_file) as f:
        #     data = f.read()
        # tokens = tagger.tag(data)
        # # super_df = pd.read_csv(super_file, sep='\t')
        # super_df = pd.read_csv(super_file, sep=',')
        # event_coref = pd.read_csv(event_coref_file)
        #
        #
        # sBytes, eBytes = [], []
        # for idx in range(super_df.shape[0]):
        #     start_token = super_df.iloc[idx]['start_token']
        #     end_token = super_df.iloc[idx]['end_token']
        #
        #     token_obj = tokens[start_token]
        #     end_token_obj = tokens[end_token]
        #     token_start_byte, token_end_byte = token_obj.startByte, end_token_obj.endByte
        #     sBytes.append(token_start_byte)
        #     eBytes.append(token_end_byte)
        # super_df['StartByte'] = sBytes
        # super_df['EndByte'] = eBytes
        # super_df.to_csv(super_file, index=None)
        #
        # event_label = []
        # for idx in range(event_coref.shape[0]):
        #     verb = event_coref.iloc[idx]['verb']
        #     verb_start_byte = event_coref.iloc[idx]['verb_start_byte_text']
        #
        #     super_line = super_df.loc[super_df['StartByte'] == verb_start_byte]
        #     if super_line.shape[0]:
        #         supersense = super_line.iloc[0]['supersense_category']
        #         print(verb,  super_line.iloc[0]['text'])
        #         assert verb in super_line.iloc[0]['text']
        #
        #         if ('verb' in supersense) and (supersense != 'verb.stative'):
        #             event_label.append(1)
        #         else:
        #             event_label.append(0)
        #     else:
        #         event_label.append(0)
        # event_coref['event_label'] = event_label
        # event_coref.to_csv(event_coref_file, index=None)
        #
        #
        #
        #
        #
        #
        #
