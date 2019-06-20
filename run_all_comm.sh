

#!/bin/bash
# Run the framework with all community detection methods for comparison

methods="community_multilevel"
#community_leading_eigenvector community_fastgreedy community_walktrap

ROOT_DIR="output_bert"
mkdir $ROOT_DIR
for val in $methods; do
    echo $val
    mkdir -p $ROOT_DIR/$val/sent_dir
	mkdir -p $ROOT_DIR/$val/net_dir
	mkdir -p $ROOT_DIR/$val/markov_dir
	mkdir -p $ROOT_DIR/$val/motif_dir

	python main.py \
    --book_list_file book_list.txt \
    --label_list_file label_list_CAT.txt \
    --book_dir data/livrosCategorias \
    --log_file $ROOT_DIR/$val/log.txt \
    --encoding_method bert \
    --word2vec_file GoogleNews-vectors-negative300.bin \
    --bert_dir bert/uncased_L-12_H-768_A-12 \
    --sent_dir $ROOT_DIR/$val/sent_dir \
    --net_dir $ROOT_DIR/$val/net_dir \
    --save_nets \
    --save_labels \
    --comm_method $val \
    --markov_dir $ROOT_DIR/$val/markov_dir \
    --save_markov \
    --range_cut_begin 0.01 \
    --range_cut_end 0.205 \
    --range_cut_step 0.005 \
    --save_motifs \
    --motif_dir $ROOT_DIR/$val/motif_dir
done