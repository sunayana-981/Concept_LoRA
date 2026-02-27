#!/bin/bash

cd ../..

# custom config
DATA=/DATA/cs22btech11005005/Concept_LoRA/Concept_LoRA/data/Caltech101
TRAINER=ZeroshotCLIP
DATASET=caltech101
CFG=vit_b16  # rn50, rn101, vit_b32 or vit_b16

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only

<<COMMENT
python train.py \
    --root /DATA/cs22btech11053/Concept_Lora/Concept_LoRA/data/OxfordIIITPet \
    --seed 1 \
    --trainer ZeroshotCLIP \
    --dataset-config-file configs/datasets/oxford_pets.yaml \
    --config-file configs/trainers/CoOp/vit_b16.yaml \
    --output-dir output/ZeroshotCLIP/vit_b16/oxford_pets/seed1 \
    --eval-only
    --config-file configs/trainers/PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx.yaml \
COMMENT