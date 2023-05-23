# fair-fairytale-nlp
Private repo for Fair Fairytale Challenge.

## 0. Create a Virtual Environment

Note: The testing python version is 3.8.13.
```
conda create --name fairy38  python==3.8.13

conda activate fairy38
```

## 1. Installation and Environment

```
pip install -r requirements.txt
```

## 2. Downloading Model
```
chmod u+x install_model.sh
./install_model.sh
```
## 3. Demo Commands

### Single Story
To run the pipeline on a single story text file (.txt), at the root dir of the project:
```
export PYTHONPATH=$(pwd)
python utils/run_bias_pipeline_single.py --story-infile /path/to/stories.txt --pipeline-dir /path/to/output/
```

You can add the following arguments as needed.
| | |Description|
|-|-|-----------|
|--story-id|-si|Story id used for saving files. If not provided, the story id will be the file name of the story infile.|
|--pipelines|-p|List of pipelines to run. Default is full pipeline. Options: full, booknlp, characters_sentences, attributes, srl, actions_args, supersense, temporal_events, temporal_events_characters|
|--pronoun-only-entities|-poe|Flag to keep character entities only referred to with pronouns.|
|--keep-all-verbs|-kav|Flag to keep stative verbs.|
### Batch Stories
To run the pipeline on multiple stories, store each story as a separate text file (.txt) in a single directory. At the root dir of the project:
```
export PYTHONPATH=$(pwd)
python utils/run_bias_pipeline_batch.py --story-dir /path/to/stories/ --pipeline-dir /path/to/output/
```

You can add the following arguments as needed.
| | |Description|
|-|-|-----------|
|--pipelines|-p|List of pipelines to run. Default is full pipeline. Options: full, booknlp, characters_sentences, attributes, srl, actions_args, supersense, temporal_events, temporal_events_characters|
|--pronoun-only-entities|-poe|Flag to keep character entities only referred to with pronouns.|
|--keep-all-verbs|-kav|Flag to keep stative verbs.|

### Fairytale Corpus

At the root dir of the project:
```
export PYTHONPATH=$(pwd)
python utils/run_FairytaleQA_bias_pipeline.py
```

## 4. CPU vs. GPU
The CPU version is expected to run very long time, approximately 20 mins on my Macbook, especially slow for TE_inference. 
