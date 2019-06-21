
#!/bin/bash
# Run the classifiers with all community detection methods for comparison

methods="community_multilevel community_leading_eigenvector community_fastgreedy community_walktrap"

#cls_results
SAVE_DIR="cls_results_freq"
mkdir $SAVE_DIR
for val in $methods; do
    echo $val
    mkdir -p $SAVE_DIR/$val
    python classification_thematic.py \
    --label_list_file label_list_CAT.txt \
    --results_dir $SAVE_DIR/$val \
    --range_cut_begin 0.01 \
    --range_cut_end 0.205 \
    --range_cut_step 0.005 \
    --motif_dir output/$val/motif_dir
done