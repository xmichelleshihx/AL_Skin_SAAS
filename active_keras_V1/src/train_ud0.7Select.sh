#!/bin/sh

python3 task1_resnet101-256-lr_second_uncertainWithaugselect.py \
    --data_path='../../../../2017_ISBI_suply/224data/task1' \
    --lr=1e-3 \
    --nb=300 \
    --batch_size=32 \
    --class_txt='../../../../2017_ISBI_suply/224data/task1.txt' \
    --output_path="../results_ud0.7/" \
    --save_select_txt_path='../select_txt_byud0.7' \
    --json_name="1429_2018_08_20_23_08_15.json" \
    # --ud_ratio=0.7 \
    # --val_model_path='../' \
    # --ud_select=True \
    # --is_save_json=True \
    # --is_first_selection=True \
    

