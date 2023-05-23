
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd
import csv
import os
import logging
import argparse
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from transformers import *
import transformers
from models import TEClassifierRoberta, TEClassifier
from optimization import *
from utils import *
from pathlib import Path
"""
Several steps to pipeline the code; 
1. For dataset curation, we can first create a folder, and put all TE_dataset into this format; 

2. In addition to pickle file data, we also include a csv file for final output format; 

3. Improve this file to load both pickle, and csv, and directly output desired output file format. 

4. Automate for large number of files. 


"""






logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

PYTORCH_PRETRAINED_ROBERTA_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_ROBERTA_CACHE',
                                                  Path.home() / '.pytorch_pretrained_roberta'))
PYTORCH_PRETRAINED_BERT_CACHE = Path(os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                               Path.home() / '.pytorch_pretrained_bert'))


class Arg():
    def __init__(self):

        self.data_dir = "./data/"
        self.model = "roberta-large"
        self.task_name ="transfer"
        #please modify the TE model to run;
        self.model_dir = "output/transfer_matres_roberta-large_batch_2_lr_5e-6_epochs_10_seed_23_1.0/"
        self.te_type = "matres"
        self.max_seq_length = 200
        self.do_lower_case = True
        self.eval_batch_size = 32
        self.mlp_hid_size = 64
        self.eval_ratio = 1.0
        self.no_cuda = False
        self.seed = 23
        self.device_num = '0'
def main():
    args = Arg()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_num

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    task_name = args.task_name.lower()

    logger.info("current task is " + str(task_name))

    if args.te_type in ['tbd']:
        label_map = tbd_label_map
    if args.te_type in ['matres']:
        label_map = matres_label_map
    if args.te_type in ['red']:
        label_map = red_label_map
    num_classes = len(label_map)

    model_state_dict = torch.load(args.model_dir + "pytorch_model.bin")
    if 'roberta' in args.model:
        tokenizer = transformers.RobertaTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
        cache_dir = PYTORCH_PRETRAINED_ROBERTA_CACHE / 'distributed_-1'
        model = TEClassifierRoberta.from_pretrained(args.model, state_dict=model_state_dict,
                                                    cache_dir=cache_dir, mlp_hid=args.mlp_hid_size,
                                                    num_classes=num_classes, return_dict=False)

    else:
        tokenizer = transformers.BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_-1'
        model = TEClassifier.from_pretrained(args.model, state_dict=model_state_dict,
                                             cache_dir=cache_dir, mlp_hid=args.mlp_hid_size,
                                             num_classes=num_classes, return_dict=False)

    model.to(device)
    #     for eval_file in ['dev', 'test']:
    for eval_file in ['old_dschang']:
        eval_features_te = convert_examples_to_features_te(args.data_dir, args.te_type, eval_file,
                                                           tokenizer, args.max_seq_length, True)


        te_sample_size = len(eval_features_te)
        logger.info("***** Running evaluation *****")
        logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info("  Num TE examples = %d", len(eval_features_te))

        eval_input_ids_te = torch.tensor(select_field_te(eval_features_te, 'input_ids'), dtype=torch.long)
        eval_input_mask_te = torch.tensor(select_field_te(eval_features_te, 'input_mask'), dtype=torch.long)
        eval_segment_ids_te = torch.tensor(select_field_te(eval_features_te, 'segment_ids'), dtype=torch.long)
        eval_lidx_s = torch.tensor(select_field_te(eval_features_te, 'lidx_s'), dtype=torch.long)
        eval_lidx_e = torch.tensor(select_field_te(eval_features_te, 'lidx_e'), dtype=torch.long)
        eval_ridx_s = torch.tensor(select_field_te(eval_features_te, 'ridx_s'), dtype=torch.long)
        eval_ridx_e = torch.tensor(select_field_te(eval_features_te, 'ridx_e'), dtype=torch.long)
        eval_pred_inds = torch.tensor(select_field_te(eval_features_te, 'pred_ind'), dtype=torch.long)
        eval_label_te = torch.tensor([f.label for f in eval_features_te], dtype=torch.long)
        eval_input_length_te = torch.tensor([f.length for f in eval_features_te], dtype=torch.long)

        eval_data = TensorDataset(eval_input_ids_te, eval_input_mask_te, eval_segment_ids_te, eval_label_te,
                                  eval_lidx_s, eval_lidx_e, eval_ridx_s, eval_ridx_e, eval_pred_inds,
                                  eval_input_length_te)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        #         if args.analyze:
        #             temp_group_inds = [f.temp_group_ind for f in eval_features_te]

        all_preds, te_preds, all_golds = [], [], []

        idx2label = {k: v for k, v in enumerate(label_map)}
        te_true_labels = [idx2label[f.label] for f in eval_features_te[:te_sample_size]]
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids_te, input_mask_te, segment_ids_te, labels_te, \
            lidx_s, lidx_e, ridx_s, ridx_e, pred_ind, length_te = batch
            with torch.no_grad():
                loss_te, logit_te = model(input_ids_te, token_type_ids_te=segment_ids_te,
                                          attention_mask_te=input_mask_te, lidx_s=lidx_s, lidx_e=lidx_e,
                                          ridx_s=ridx_s, ridx_e=ridx_e, length_te=length_te,
                                          labels_te=labels_te)

                logit_te = logit_te.detach().cpu().numpy()
                pred_te = np.argmax(logit_te, axis=1).tolist()
                te_preds.extend(pred_te)

        te_preds_labels = [idx2label[x] for x in te_preds[:te_sample_size]]

        output = pd.read_csv('data/matres/old_dschang_temp.csv')
        output['te_rel'] = te_preds_labels
        output.to_csv('data/matres/old_dschang'+'_output'+ '.csv', index=None)

        # open the file in the write mode
        with open('./test_data/preds_' + eval_file + '.csv', 'w') as f:
            # create the csv writer
            writer = csv.writer(f)
            header = ['true', 'pred']
            # write a row to the csv file
            writer.writerow(header)
            for i in range(len(te_true_labels)):
                writer.writerow([te_true_labels[i], te_preds_labels[i]])

        report = ClassificationReport(args.task_name + "-" + args.te_type, te_true_labels, te_preds_labels)
        print(report)

if __name__ == "__main__":
    main()