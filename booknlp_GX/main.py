#you can run the following code to run the boookNLP code on every story in the split_by_origin 
#folder of the FairytaleQA_Dataset;
#it will create a result folder in the same directory as the story file. 



import os
from booknlp.booknlp import BookNLP
from tqdm import tqdm
import glob
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

class FairyBookNLP:
    def __init__(self):
        self.model_parems = \
            {
            "pipeline":
                "coref,entity,quote,supersense,event"
            , "model": "big"
        }

        self.booknlp = BookNLP("en", self.model_parems)

    def process(self, input_file, output_dir=None):
        """
        specify an output directory to create the story folder and store all booknlp files inthe story folder
        Leave empty if to create output folder in the same directory as the txt story file
        """
        story_name = input_file.split('/')[-1]

        story_name = story_name.split('.')[0]


        output_dir = os.path.join(output_dir, story_name)

        # output_directory = os.path.join(output_path, story_name)
        try:
            self.booknlp.process(input_file, output_dir, story_name)
        except:
            print("failed to process one story: ", story_name)

    def process_split_by_origin(self, data_dir):
        """
        data_dir is the path to the split_by_origin directory, and only txt files are expected in the stories

        #Note, please use global path for this data_dir, not local dir;
        eg. use sth like '/home/gxu21/fair-fairytale-nlp/booknlp/FDText'
        """
        path = data_dir
        origin_ls = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        # print(origin_ls)
        for idx, origin in tqdm(enumerate(origin_ls)):
            origin_path = os.path.join(path, origin)
            stories = os.listdir(origin_path)
            stories = [x for x in stories if '.txt' in x]
            # stories = glob.glob(origin_path)
            print(stories)
            for story in tqdm(stories):
                story_path = os.path.join(origin_path, story)
                # Input file to process
                input_file = story_path
                story_folder = story_path.split(".")[0]

                isExist = os.path.exists(story_folder)

                if not isExist:
                    # Create a new directory because it does not exist
                    os.makedirs(story_folder)

                # Output directory to store resulting files in
                output_directory = story_folder
                # File within this directory will be named ${book_id}.entities, ${book_id}.tokens, etc.
                # name = "".join(model_params["pipeline"])
                book_id = input_file.split(".")[0].split('/')[-1]
                try:
                    self.booknlp.process(input_file, output_directory, book_id)
                except:
                    print("failed to process one story: ", story)
                    pass

            print("orgin processed: ", origin)


