# Complex Question Decomposition for Semantic Parsing

This is the code base for ACL'19 paper `Complex Question Decomposition for Semantic Parsing`. 

## 1. Preprocess

### 1.1 Download raw data

Download ComplexWebQ data, prepare environment and libraries. 

### 1.2 Requirements for preprocess

In order to run preprocess, you should put the following files in DATA_PATH directory, DATA_PATH is defined in the script.
- ComplexWebQuestions_train.json
- ComplexWebQuestions_dev.json
- ComplexWebQuestions_test.json (we need any other information in above files)
- train.json, dev.json, test.json (we need the splitted sub questions in these files, it is not included in raw data, 
but we generate and prepare them for users in `complex_questions` directory, also you can generate them by following steps)

### 1.2.1 Generate golden sub question split points by yourself

- `cd WebAsKB`.
- Prepare a StanfordCoreNLP server in localhost following 1.2.2.
- Change data_dir setting in `WebAsKB/config.py`.
- Change EVALUATION_SET setting in `WebAsKB/config.py` to train, dev and test, and Run `python webaskb_run.py gen_golden_sup` for three times.
- By this time, you can get `train.json, dev.json, test.json` in DATA_PATH.

### 1.2.2 Prepare a StanfordCoreNLP server

In order to run the POS annotation process, you should download and start a StanfordCoreNLP server in localhost:9003.
- Download from `https://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip`, unzip and cd to it.
- Start server using `java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9003 -timeout 15000`.

### 1.3 Modify script

We provide a template script `scripts/run.sh` for the users, and you need to change the following directory settings at least to run it.

- DATA_PATH: where the data root directory is.
- RUN_T2T: the root directory of the code base. 

### 1.3 Run Preprocess

Now run `scripts/run.sh preprocess`, the command will generate the data format for our model, and annotate POS labels.

## 2. Prepare

Prepare Glove pretrained embedding file `glove.6B.300d.txt` and put it in DATA_PATH/embed/.

`scripts/run.sh prepare` will shuffle the dataset and build vocabulary file.

## 3. Train

To train our decompose model, use `scripts/run.sh train`.

To train our semantic parsing model, use `scripts/run.sh train_lf`.

## 4. Test

`scripts/run.sh test`, it will generate decomposed query with a input file, and print bleu-4 & rouge-l score compared to references.

`scripts/run.sh test_lf`, it will generate logical form with a input file, and print EM score compared to references. 

## Citation

If you use this code in your research, please kindly cite our paper via the following BibTeX.

```bibtex
@inproceedings{Zhang2019HSP,
author = {Zhang, Haoyu and Cai, Jingjing and Xu, Jianjun and Wang, Ji},
booktitle = {Conference of the Association for Computational Linguistics (ACL)},
year = {2019}
}

```
