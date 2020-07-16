#!/bin/sh

python3 task1_resnet101-256-lr_second_uncertainWithaugselect.py \
    --data_path='../../../../2017_ISBI_suply/224data/task1' \
    --lr=1e-3 \
    --nb=150 \
    --batch_size=32 \
    --class_txt='../../../../2017_ISBI_suply/224data/task1.txt' \
    --output_path="../results_udWithaug0.7/" \
    --save_select_txt_path='../select_txt_byud0.7' \
    --json_name="2143_2018_08_21_15_18_44.json" \
    --is_GILP="True" \
    # --ud_ratio=0.7 \
    # --ud_select=True \
    # --val_model_path='../results_udWithaug0.7/2018_08_21_09_56_17/clean_2018_08_21_09_56_17_baseline_weightsS.hdf5' \
    # --is_save_json=True \
    # --is_first_selection=True \
    
    

