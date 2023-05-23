import os

from tqdm import tqdm
import pandas as pd

from booknlp_GX.booknlp.booknlp import BookNLP

class BookNLPPipeline:
    def __init__(self, story_id, story_file, bookNLP_model, pipeline_dir):
        self.story_id = story_id
        self.story_file = story_file

        self.output_story_dir = self.get_story_output_dir(pipeline_dir)       

        # Run model
        try:
            bookNLP_model.process(self.story_file, self.output_story_dir, self.story_id)
        except:
            print("Failed to process story: {}".format(self.story_id))

        story_dir_with_prefix = os.path.join(self.output_story_dir, story_id)

        tokens_file = story_dir_with_prefix + '.tokens'
        entities_file = story_dir_with_prefix + '.entities'
        character_meta_file = story_dir_with_prefix + '.character_meta_pron'
        supersense_file = story_dir_with_prefix + '.supersense'

        self.tokens_df = pd.read_csv(tokens_file, sep='\t')
        self.entities_df = pd.read_csv(entities_file, sep='\t')
        self.character_meta_df = pd.read_csv(character_meta_file, sep='\t')
        self.supersense_df = pd.read_csv(supersense_file, sep='\t')

    def get_story_output_dir(self, pipeline_dir):
        output_story_dir = os.path.join(pipeline_dir, self.story_id)
        output_story_dir_exists = os.path.exists(output_story_dir)

        if not os.path.exists(output_story_dir):
            os.makedirs(output_story_dir)

        return output_story_dir

def load_BookNLP_model():
    # Set GPU Environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # See Issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"

    model_params = {
        "pipeline": "coref,entity,quote,supersense",
        "model": "big"
        }

    return BookNLP("en", model_params)
