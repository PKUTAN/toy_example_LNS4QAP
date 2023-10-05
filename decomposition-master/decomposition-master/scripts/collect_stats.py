import sys
sys.path.append('.')

from absl import flags
from absl import app

FLAGS = flags.FLAGS

import numpy as np
import os

flags.DEFINE_string('stat_dir', None, 'stat directory')
flags.DEFINE_integer('min_k', 1, 'Min k')
flags.DEFINE_integer('max_k', 5, 'Max k')
flags.DEFINE_integer('min_t', 1, 'Min t')
flags.DEFINE_integer('max_t', 3, 'Max t')
flags.DEFINE_integer('steps', 1, '')
flags.DEFINE_string('mode', 'random_stats', '')


def collect_random_stats(stat_dir, min_k, max_k,
                         min_t, max_t, steps, init_time=3):
    mean_stat = np.zeros((max_k-min_k+1, max_t-min_t+1))
    std_stat = np.zeros((max_k-min_k+1, max_t-min_t+1))    
    
    for k in range(min_k, max_k+1):
        for t in range(min_t, max_t+1):
            output = os.path.join(stat_dir,
                                  'random_k_%d_step_%d_time_%d_init_%d'
                                  %(k, steps, t, init_time),
                                  'stats.txt')
            with open(output, 'r') as f:
                lines = f.readlines()

            std = float(lines[-1].strip().rsplit(maxsplit=1)[1])
            mean = float(lines[-2].strip().rsplit(maxsplit=1)[1])
            mean_stat[k-min_k][t-min_t] = mean
            std_stat[k-min_k][t-min_t] = std

    for k in range(max_k+1-min_k):
        for t in range(max_t+1-min_t):
            print('%.2f +/- %.2f' %(mean_stat[k][t], std_stat[k][t]), end=' ')
        print()
            

def main(argv):
    stat_dir = FLAGS.stat_dir
    min_k = FLAGS.min_k
    max_k = FLAGS.max_k
    min_t = FLAGS.min_t
    max_t = FLAGS.max_t
    mode = FLAGS.mode
    steps = FLAGS.steps

    if mode == 'random_stats':
        collect_random_stats(stat_dir, min_k, max_k,
                             min_t, max_t, steps)


if __name__ == "__main__":
    app.run(main)
