# SpERT: Span-based Entity and Relation Transformer

CUDA_VISIBLE_DEVICES=1 \
python spert.py train --config configs/vabert_train_eyebert.conf

python scripts/make_data_from_labelstudio_joint_training.py \
       --data_dir "/mnt/data/quangng/vabert/data/v1.2" \
       --output_dir "data/datasets/vabert_eyebert" \
       --tokenizer_name "/mnt/data/quangng/vabert/pretrain/models/EyeBERTv10/weights/checkpoint-100000"

python scripts/make_data_from_labelstudio_joint_training.py \
       --data_dir "/mnt/data/quangng/vabert/data/v1.2" \
       --output_dir "data/datasets/vabert_biobert" \
       --tokenizer_name "dmis-lab/biobert-v1.1"


export CUDA_VISIBLE_DEVICES=0
python spert.py eval --config configs/vabert_eval_eyebert.conf

export CUDA_VISIBLE_DEVICES=2
python spert.py eval --config configs/vabert_eval_biobert.conf