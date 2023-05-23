
"""
This is the main file to run the Temporal Relation Extraction Pipeline

Before running this file, we assume we have
1. story.txt
2. story.sentences.csv
3. story.supersense
4. story.verb_subj_dobj.srl.csv

under the same directory;

The output of this pipeline will be story folders, which outputs the TE_output.csv
"""
from os import path

from newECON.code.create_event_coref import append_supersense
from newECON.code.TE_data import get_data
from newECON.code.TE_inference_lib import TE_infer


class TemporalEventsModel():
    def __init__(self):
        pass


    def check_core_components(self, dir, stories):
        """
            1. story.txt
            2. story.sentences.csv
            3. story.supersense
            4. story.verb_subj_dobj.srl.csv
            5. story.srl.json (optional, if not having this, set get_srl=True in get_TE_daa)

            must exist in dir, for each story to run
        """

        for story in stories:
            try:
                assert path.isfile(path.join(dir, story +'.txt'))
                assert path.isfile(path.join(dir, story +'.sentences.csv'))
                assert path.isfile(path.join(dir, story +'.supersense'))
                assert path.isfile(path.join(dir, story +'.actions_args.csv'))

            except:
                print('files not complete, please check which of the input file is not included')
        print('success, all component files are in-place')


    def incorporate_supersense(self, dir, story):
        """
        :param dir: the dir underwhich which expect 3 files,
                1. story.txt
                2. story.supersense
                3. story.verb_subj_dobj.srl.csv
        :param story: the name of the story to process
        :return: None;
                it updates the story.verb_subj_dobj.srl.csv file with an additional supersense column
        """
        append_supersense(dir, story)

    def get_TE_data(self, dir, story, get_srl=False,major_event=False):
        """
        :param dir: data dir
        :param story: story-name
        :param get_srl:
                        True if you want to create new srl files given the sentences file;
                        False if the srl file are already available and in-place.
        :return: TE_data instance to run TE inference; data.pickle + TE_output.csv(unpredicted)
        """
        get_data(dir, story, get_srl=get_srl,major_event=major_event)



    def run_TE_inference(self, story_dirs, model_dir=None):
        """
        The main TE inference funciton; takes the directory of a story folder, and make prediction in TE_output.csv
        :param story_dir:
                story.srl.json(optionally) srl results out of allennlp srl model
                data.pickle; the raw input data file, output by get_TE_data function
                TE_output.csv is also a raw input file, prepared by get_TE_data function
        :param model_dir:
                the directory where the inference model is saved;
        :return:
        """
        if model_dir:
            TE_infer(story_dirs, model_dir)
        else:
            TE_infer(story_dirs)


if __name__ == "__main__":
    """
    This is example usage of the pipeline running on the ../may18_data
    """
    TE_model = TemporalEventsModel()
    stories = ['ali-baba-and-forty-thieves', 'old-dschang','cinderella-or-the-little-glass-slipper'
        ,'bamboo-cutter-moon-child','leelinau-the-lost-daughter','the-dragon-princess']

    dir = '../may18_data/'
    TE_model.check_core_components(dir, stories)

    for i, story in enumerate(stories):
        TE_model.incorporate_supersense(dir, story)
        TE_model.get_TE_data(dir, story, get_srl=False)

    print('sucess curating the raw input data')

    story_dirs = [dir+x+'/' for x in stories]

    #specify here model_dir if you downloaded your model in a new destination other than
    # "../output/transfer_matres_roberta-large_batch_2_lr_5e-6_epochs_10_seed_23_1.0/"
    TE_model.run_TE_inference(story_dirs)









