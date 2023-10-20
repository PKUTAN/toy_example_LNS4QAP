#!/bin/sh

python -W ignore utils/train.py \
       --network=100,300,2 \
       --dropout=0.0 \
       --lr=0.001 \
       --weight_decay=5e-4 \
       --K=5 \
       --temp=10 \
       --p_type=maxcut \
       --steps=5 \
       --iterations=10 \
       --time_limit=1 \
       --n_traj=5 \
       --kmeans_iter=1 \
       --algo=reinforce \
       --dir_problems=data/maxcut/erdos_renyi/500_500/lpfiles/labeled \
       --mode=train \
       --model_dir=models/maxcut/erdos_renyi/500_500/rl



python -W ignore utils/train.py \
       --network=60,300,200,2 \
       --dropout=0.0 \
       --lr=0.001 \
       --weight_decay=5e-4 \
       --K=5 \
       --temp=10 \
       --p_type=qap \
       --steps=4 \
       --iterations=10 \
       --time_limit=5 \
       --n_traj=5 \
       --kmeans_iter=1 \
       --algo=reinforce \
       --dir_problems=data/qap/erdos_renyi/500_500/lpfiles/labeled \
       --mode=train \
       --model_dir=models/qap/erdos_renyi/500_500/rl
