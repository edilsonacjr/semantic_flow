

#!/bin/bash
# Run the framework with all community detection methods for comparison

methods="community_multilevel community_leading_eigenvector community_fastgreedy community_walktrap"

mkdir output
for val in $methods; do
    echo $val
    mkdir -p output/$val/sent_dir
	mkdir -p output/$val/net_dir
	mkdir -p output/$val/markov_dir
	mkdir -p output/$val/motif_dir

	python main.py \
    --book_list_file book_list.txt \
    --label_list_file label_list_CAT.txt \
    --book_dir data/livrosCategorias \
    --log_file output/$val/log.txt \
    --word2vec_file GoogleNews-vectors-negative300.bin \
    --sent_dir output/$val/sent_dir \
    --net_dir output/$val/net_dir \
    --save_nets \
    --save_labels \
    --comm_method $val \
    --markov_dir output/$val/markov_dir \
    --save_markov \
    --range_cut_begin 0.01 \
    --range_cut_end 0.205 \
    --range_cut_step 0.005 \
    --save_motifs \
    --motif_dir output/$val/motif_dir
done