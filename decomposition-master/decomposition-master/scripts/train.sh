#!/bin/sh

# for i in {0..4}
# do
#     python algo/supervise.py --lp_dir=/tmp/jssong/mvc_weighted/barabasi_albert/4000_4000/lpfiles/labeled --mode=train --p_type=mvc --model_file=/tmp/jssong/models/mvc_weighted/barabasi_albert/4000_200/300_100_8_5000 --network=300,100,8 --cluster_file=/tmp/jssong/mvc_weighted/barabasi_albert/random_logs/4000_4000/labeled/random_k_8_step_1_time_5/best_clusters.pkl --epoch=5000 -seed=$i --time_limit=5
# done



# for i in {0..4}
# do
#     python algo/supervise.py --mode=train --lp_dir=data/maxcut/erdos_renyi/new_instances/500_100/lpfiles/train/ --cluster_file=data/maxcut/erdos_renyi/new_instances/500_100/random_logs/train/random_k_5_step_1_time_1/best_clusters.pkl --epoch=500 --network=300,100,5 --model_file=/tmp/jssong/models/maxcut/erdos_renyi/500_500/300_100_5_500 --seed=$i --time_limit=1
# done


# for i in {0..4}
# do
#     python algo/supervise.py --mode=train --lp_dir=data/maxcut/barabasi_albert/new_instances/500_25/lpfiles/train/ --cluster_file=data/maxcut/barabasi_albert/new_instances/500_25/random_logs/train/random_k_5_step_1_time_1/best_clusters.pkl --epoch=500 --network=300,100,5 --model_file=/tmp/jssong/models/maxcut/barabasi_albert/500_25/300_100_5_500 --seed=$i --time_limit=1
# done

# CATS arbitrary 2000 BC

# for i in 861 6046 369 213 8541
# do
#     python algo/supervise.py --mode=train --lp_dir=data/cats/arbitrary_lp/2000_4000/train --cluster_file=random_logs/data/cats/arbitrary_lp/2000_4000/train/random_k_2_step_10_time_1/best_clusters.pkl  --epoch=500 --network=100,300,2 --model_dir=/tmp/jssong/models/cats/arbitrary/2000_4000/bc --seed=$i --time_limit=1 --p_type=cats
# done

# CATS arbitrary 2000 FT

for i in 861 6046 369 213 8541
do
    python algo/forward_training.py --mode=train --lp_dir=data/cats/arbitrary_lp/2000_4000/train --cluster_file=random_logs/data/cats/arbitrary_lp/2000_4000/train/random_k_2_step_10_time_1/best_clusters.pkl  --epoch=500 --network=100,300,2 --model_dir=/tmp/jssong/models/cats/arbitrary/2000_4000/ft --seed=$i --time_limit=1 --p_type=cats --steps=10
done

# python -W ignore utils/train.py \
#        --network=10,50,9 \
#        --dropout=0.0 \
#        --lr=0.001 \
#        --weight_decay=5e-4 \
#        --K=5 \
#        --temp=10 \
#        --p_type=maxcut \
#        --steps=1 \
#        --iterations=10 \
#        --time_limit=1 \
#        --n_traj=8 \
#        --kmeans_iter=1 \
#        --algo=reinforce \
#        --dir_problems=data/maxcut/barabasi_albert/500_25/lpfiles/test \
#        --dir_graphs=data/maxcut/barabasi_albert/500_25/gpickle/test \
#        --mode=train \
#        --model_file=data/maxcut/barabasi_albert/500_25/lpfiles/test_train_logs/2020_01_09_16_53_47/models/iter_6.pt


#       --dir_problems=data/maxcut/barabasi_albert/500_25/lpfiles/valid \
#       --dir_graphs=data/maxcut/barabasi_albert/500_25/gpickle/valid \

#       --model_file=data/maxcut/barabasi_albert/500_25/lpfiles/test_train_logs/2020_01_07_16_47_59/models/iter_9.pt \


#       --dir_problems=data/maxcut/erdos_renyi/500_500/lpfiles/valid \
#       --dir_graphs=data/maxcut/erdos_renyi/500_500/gpickle/valid \


#       --dir_problems=data/maxcut/barabasi_albert/300_15_single/lpfiles \
#       --dir_graphs=data/maxcut/barabasi_albert/300_15_single/gpickle



#       --dir_problems=data/maxcut/erdos_renyi/200_single/lpfiles/ \
#       --dir_graphs=data/maxcut/erdos_renyi/200_single/gpickle/ \
