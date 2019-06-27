# Semantic Flow in Language Networks

This repository provides the implementation (Python) of the Semantic Flow framework described in [[1]](#Semantic-flow-in-language-networks). All data used for the paper is available in [here]().


## References

Please cite [[1]](#Semantic-flow-in-language-networks) if using this code.


[1] Edilson A. Corrêa Jr, Vanessa Q. Marinho, Diego R. Amancio, [*Semantic flow in language networks*](https://arxiv.org/abs/1905.07595)

```
@article{correa2019semantic,
  title={Semantic flow in language networks},
  author={Corr{\^e}a Jr, Edilson A and Marinho, Vanessa Q and Amancio, Diego R},
  journal={arXiv preprint arXiv:1905.07595},
  year={2019}
}
```

## Tutorial

# 1. Setup env

First you need to download and install Anaconda, a tutorial can be found [here](https://docs.anaconda.com/anaconda/install/). Then just setup a new enviroment this project:

```
git clone https://github.com/edilsonacjr/semantic_flow.git
cd semantic_flow
conda env create -f environment.yaml
```

Obs.: the last line of the environment.yaml needs to be edited if Anaconda was installed in a non conventional way.


# 2. Download Word2Vec and BERT

Here we choose to use the pre-trained word2vec made available by Mikolov ([link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)).

For BERT encoding first we get the code and then a pre-trained model
```
git clone https://github.com/google-research/bert.git
cd bert 
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```

# 3. Run Semantic flow

Now that you have a working environmente its possible to run the Semantic Flow framework.

Running with Word2Vec encoding:

```
ROOT_DIR="output"
mkdir $ROOT_DIR
EXPERIMENT="test1"
mkdir -p $ROOT_DIR/$EXPERIMENT/sent_dir
mkdir -p $ROOT_DIR/$EXPERIMENT/net_dir
mkdir -p $ROOT_DIR/$EXPERIMENT/markov_dir
mkdir -p $ROOT_DIR/$EXPERIMENT/motif_dir

python main.py \
--book_list_file book_list.txt \
--label_list_file label_list_CAT.txt \
--book_dir data/livrosCategorias \
--log_file $ROOT_DIR/$EXPERIMENT/log.txt \
--encoding_method word2vec \
--word2vec_file GoogleNews-vectors-negative300.bin \
--sent_dir $ROOT_DIR/$EXPERIMENT/sent_dir \
--net_dir $ROOT_DIR/$EXPERIMENT/net_dir \
--save_nets \
--save_labels \
--comm_method community_multilevel \
--markov_dir $ROOT_DIR/$EXPERIMENT/markov_dir \
--save_markov \
--range_cut_begin 0.01 \
--range_cut_end 0.205 \
--range_cut_step 0.005 \
--save_motifs \
--motif_dir $ROOT_DIR/$EXPERIMENT/motif_dir
```


Running with BERT encoding:
```
ROOT_DIR="output_bert"
EXPERIMENT="test2"
mkdir $ROOT_DIR
mkdir -p $ROOT_DIR/$EXPERIMENT/sent_dir
mkdir -p $ROOT_DIR/$EXPERIMENT/net_dir
mkdir -p $ROOT_DIR/$EXPERIMENT/markov_dir
mkdir -p $ROOT_DIR/$EXPERIMENT/motif_dir

python main.py \
--book_list_file book_list.txt \
--label_list_file label_list_CAT.txt \
--book_dir data/livrosCategorias \
--log_file $ROOT_DIR/$EXPERIMENT/log.txt \
--encoding_method bert \
--bert_dir bert/uncased_L-12_H-768_A-12 \
--sent_dir $ROOT_DIR/$EXPERIMENT/sent_dir \
--net_dir $ROOT_DIR/$EXPERIMENT/net_dir \
--save_nets \
--save_labels \
--comm_method community_multilevel \
--markov_dir $ROOT_DIR/$EXPERIMENT/markov_dir \
--save_markov \
--range_cut_begin 0.01 \
--range_cut_end 0.205 \
--range_cut_step 0.005 \
--save_motifs \
--motif_dir $ROOT_DIR/$EXPERIMENT/motif_dir
```

# 4. Classification using extracted features (motifs)

Here we execyte the classification process used [[1]](#Semantic-flow-in-language-networks).

```
SAVE_DIR="cls_results"
EXPERIMENT="test1"
ROOT_DIR="output"


mkdir $SAVE_DIR

echo $val
mkdir -p $SAVE_DIR/$EXPERIMENT
python classification_thematic.py \
--label_list_file label_list_CAT.txt \
--results_dir $SAVE_DIR/$EXPERIMENT \
--range_cut_begin 0.01 \
--range_cut_end 0.205 \
--range_cut_step 0.005 \
--motif_dir $ROOT_DIR/$EXPERIMENT/motif_dir
```

Scripts for automatic comparison of community detection methods are also provided (run_all_comm.sh and run_cls_comp.sh).


For more information, you can contact me via edilsonacjr@gmail.com or edilsonacjr@usp.br.


Best, Edilson A. Corrêa Jr.




