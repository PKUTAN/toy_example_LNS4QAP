#!/bin/sh

for i in {0..4}
do
    python algo/supervise.py --lp_dir=data/cats/regions_lp/2000_4000/valid --mode=evaluate --model_file=/tmp/jssong/models/cats/regions/2000_4000/1000_300_100_2_500_seed_$i.pt --p_type=cats --cache_dir=data/cats/arbitrary_lp/random_logs/2000_4000/test/random_k_2_step_1_time_2/ --time_limit=2 --steps=5
done

# for i in {0..4}
# do
#     python algo/supervise.py --lp_dir=data/cats/arbitrary_lp/2000_4000/valid --mode=evaluate --model_file=/tmp/jssong/models/cats/arbitrary/2000_4000/1000_300_100_2_500_seed_$i.pt --p_type=cats --cache_dir=data/cats/arbitrary_lp/random_logs/2000_4000/test/random_k_2_step_1_time_2/ --time_limit=2 --steps=5
# done


# for i in {0..4}
# do
#     python algo/supervise.py --lp_dir=/tmp/jssong/mvc_weighted/barabasi_albert/4000_4000/lpfiles/valid --mode=evaluate --model_file=/tmp/jssong/models/mvc_weighted/barabasi_albert/4000_200/300_100_8_500_seed_$i.pt --p_type=mvc --cache_dir=data/cats/arbitrary_lp/random_logs/2000_4000/test/random_k_2_step_1_time_2/ --time_limit=5
# done


# for i in {0..4}
# do
#     python algo/supervise.py --lp_dir=data/maxcut/erdos_renyi/500_500/lpfiles/valid --mode=evaluate --model_file=/tmp/jssong/models/maxcut/erdos_renyi/500_500/300_100_5_500_seed_$i.pt --time_limit=1
# done


# for i in {0..4}
# do
#     python algo/supervise.py --lp_dir=data/maxcut/barabasi_albert/500_25/lpfiles/valid --mode=evaluate --model_file=/tmp/jssong/models/maxcut/barabasi_albert/500_25/300_100_5_500_seed_$i.pt --time_limit=1 --steps=5
# done
