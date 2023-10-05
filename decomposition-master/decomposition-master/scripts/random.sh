#!/bin/sh

python scripts/run_eval.py \
       --min_k=5 \
       --max_k=5 \
       --mode=random_stats \
       --p_type=maxcut \
       --time_limit=1 \
       --steps=5 \
       --iterations=8 \
       --lp_dir=data/maxcut/barabasi_albert/500_25/lpfiles/test \
       --output_dir=data/maxcut/barabasi_albert/500_25/random_logs/test_single_component

#       --lp_dir=data/maxcut/erdos_renyi/500_single/lpfiles/ \


       # --output_dir=data/maxcut/barabasi_albert/500_25/random_logs/valid_1
       
#       --lp_dir=data/maxcut/erdos_renyi/500_500/lpfiles/valid \
#       --output_dir=data/maxcut/erdos_renyi/500_500/random_logs/valid_1

       
       # --lp_dir=data/maxcut/barabasi_albert/300_15_single/lpfiles \
       # --output_dir=data/maxcut/barabasi_albert/300_15_single/random_logs

       # --lp_dir=data/maxcut/erdos_renyi/200_single/lpfiles/input10.lp       
       # --lp_dir=data/maxcut/erdos_renyi/300_300/lpfiles/labeled \
       # --output_dir=data/maxcut/erdos_renyi/300_300/random_logs_k
