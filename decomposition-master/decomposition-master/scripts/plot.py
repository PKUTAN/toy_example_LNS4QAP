import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pickle
import os
import yaml
import gurobipy

from absl import flags
from absl import app

from utils import parser
from utils import common


FLAGS = flags.FLAGS

flags.DEFINE_string('logs_dir', None, 'The log directory.')
flags.DEFINE_string('plt_title', None, 'The title of the plot.')
flags.DEFINE_integer('min_k', 2, 'The minimum number of clusters.')
flags.DEFINE_integer('max_k', 2, 'The maximum number of clusters.')
flags.DEFINE_integer('steps', 1, 'The number of steps taken.')
flags.DEFINE_string('bc_dir', '', 'The log directory.')
flags.DEFINE_string('ft_dir', '', 'The log directory.')
flags.DEFINE_string('random_dir', '', 'The log directory.')
flags.DEFINE_string('gurobi_dir', '', 'The log directory.')
flags.DEFINE_string('plt_name', None, 'The log directory.')


def plot_random_decomps_stats(logs_dir, plt_title):
    aggregate = 0
    ctr = 0
    
    for f_name in os.listdir(logs_dir):
        abs_path = os.path.join(logs_dir, f_name)
        diff_data = parser.parse_random_decomp_log(abs_path)
        aggregate += np.array(diff_data)
        ctr += 1

    aggregate /= ctr
    plt.hist(aggregate, bins='auto')
    plt.axvline(np.mean(aggregate), color='k', linestyle='dashed', linewidth=1)
    plt.title(plt_title)
    plt.show()
    print('mean: ', np.mean(aggregate))
    print('best: ', max(aggregate))

    
def visualize(lp_file, num_waypoint, num_obst, obst_file, save_f, best_cluster_f, time_limit):
    with open(best_cluster_f, 'rb') as f:
        best_clusters = pickle.load(f)

    lp_prefix = lp_file.rsplit('/', maxsplit=1)[1].split('.')[0]
    best_cluster, best_sol = best_clusters[lp_prefix]
    model = gurobipy.read(lp_file)
    model.setParam('MIPFocus', 1)
    model.setParam('TimeLimit', time_limit)

    var_dict = common.generate_var_dict(model, p_type='psulu')
    ctr = 0

    for cluster, sol in zip(best_cluster, best_sol[1:]):
        sol_dict = common.vec_to_dict(sol, var_dict)
        fixed_model = common.fix_integer_values(model.copy(), sol_dict)
        fixed_model.optimize()
        plot_psulu_decomps(fixed_model, num_waypoint, num_obst,
                           obst_file, save_f+'%d.png' %ctr, cluster[1])
        ctr += 1

    
def plot_psulu_decomps(model, num_waypoint, num_obst, obs_file, save_f, cluster):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    
    with open(obs_file, 'r') as f:
        obs = yaml.load(f, Loader=yaml.FullLoader)

    obs_dict = obs['environment']['obstacles']
    centers = {}
    waypoints = {}
    
    for i in range(len(obs_dict)):
        i_obs = obs_dict['obs_%d' %i]
        corners = np.stack(i_obs['corners'], axis=0)
        ax.add_patch(patches.Polygon(corners, alpha=0.5))
        centers[i] = np.mean(corners, axis=0)

    xs = []
    ys = []
    
    for i in range(num_waypoint):
        x_var = model.getVarByName('x_%d_0' %i)
        y_var = model.getVarByName('x_%d_1' %i)        
        x_pos = x_var.X
        y_pos = y_var.X
        ax.plot(x_pos, y_pos, 'kx')
        waypoints[i] = [x_pos, y_pos]

    obs_ctr = dict([(i, 0) for i in range(num_obst)])
    wp_ctr = dict([(i, 0) for i in range(num_waypoint)])
    for idx in cluster:
        idx_wp = idx // num_obst
        idx_obs = idx % num_obst
        obs_ctr[idx_obs] += 1
        wp_ctr[idx_wp] += 1
        center = centers[idx_obs]
        waypoint = waypoints[idx_wp]
        # plt.plot([center[0], waypoint[0]], [center[1], waypoint[1]], linewidth=1)

    top_k = 5
    sorted_obs = sorted(obs_ctr.items(), key=lambda x: x[1], reverse=True)
    sorted_wp = sorted(wp_ctr.items(), key=lambda x: x[1], reverse=True)
    
    for i in range(top_k):
        idx_obs = sorted_obs[i][0]
        idx_wp = sorted_wp[i][0]
        center = centers[idx_obs]
        waypoint = waypoints[idx_wp]
        plt.plot(center[0], center[1], 'ro')
        plt.plot(waypoint[0], waypoint[1], 'bs')        

    plt.savefig(save_f, bbox_inches='tight')


def plot_obj_time(bc_dir, ft_dir, random_dir, gurobi_dir, plt_name):

    with open(bc_dir, 'r') as f:
        lines = f.readlines()
        bc_objs = [float(x) for x in lines[0].split()]
        bc_times = [float(x) for x in lines[1].split()]

    with open(ft_dir, 'r') as f:
        lines = f.readlines()
        ft_objs = [float(x) for x in lines[0].split()]
        ft_times = [float(x) for x in lines[1].split()]

    with open(random_dir, 'r') as f:
        lines = f.readlines()
        random_objs = [float(x) for x in lines[1].split()]
        random_times = [float(x) for x in lines[2].split()]
        if 'psulu' in random_dir and '40_40' in random_dir:
            for i, t in enumerate(random_times):
                if i > 0:
                    random_times[i] += random_times[i] + random_times[0]

    with open(gurobi_dir, 'r') as f:
        lines = f.readlines()
        gurobi_objs = []
        gurobi_times = []

        for line in lines:
            t, obj = line.split()
            gurobi_objs.append(float(obj))
            gurobi_times.append(float(t))

        if 'cats' in gurobi_dir and 'arbitrary' in gurobi_dir:
            gurobi_objs.append(gurobi_objs[-1])
            gurobi_times.append(7.0)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if 'psulu' in bc_dir:
        random_times[0] = 0.0
        bc_times[0] = 0.0
        ft_times[0] = 0.0

    #random_times = [x+1 for x in random_times]
    #bc_times = [x+1 for x in random_times]
    #ft_times = [x+1 for x in random_times]    
    #gurobi_times = [x+1 for x in gurobi_times]
    
    plt.step(random_times, random_objs, label='Random-LNS', where='post')
    plt.step(bc_times, bc_objs, label='BC-LNS', where='post')
    plt.step(ft_times, ft_objs, label='FT-LNS', where='post')
    plt.step(gurobi_times, gurobi_objs, label='Gurobi', where='post')
    ax.set_title('Objective Values with Wall-clock Time')
    ax.set_xlabel('Wall-clock Time (s)')
    ax.set_ylabel('Objective')
    ax.legend()

    plt.savefig(plt_name, bbox_inches='tight')


def plot_obj_step(bc_dir, ft_dir, random_dir):
    mean_bc_objs = []
    mean_ft_objs = []
    mean_random_objs = []
    
    for f_name in os.listdir(bc_dir):
        bc_log = os.path.join(bc_dir, f_name)
        ft_log = os.path.join(ft_dir, f_name)
        random_log = os.path.join(random_dir, f_name)

        with open(bc_log, 'r') as f:
            lines = f.readlines()
            bc_objs = [float(x) for x in lines[0].split()]

        with open(ft_log, 'r') as f:
            lines = f.readlines()
            if len(lines) < 1:
                continue
            ft_objs = [float(x) for x in lines[0].split()]

        with open(random_log, 'r') as f:
            lines = f.readlines()
            random_objs = []
            ctr = 0

            for idx, line in enumerate(lines):
                if idx % 3 == 1:
                    ctr += 1
                    objs = [float(x) for x in line.split()]
                    if len(random_objs) == 0:
                        random_objs = objs[:]
                    else:
                        random_objs = [x+y for x,y in zip(random_objs, objs)]

            random_objs = [x / ctr for x in random_objs]

        mean_bc_objs.append(bc_objs)
        mean_ft_objs.append(ft_objs)
        mean_random_objs.append(random_objs)

    mean_bc_objs = np.mean(mean_bc_objs, axis=0)
    mean_ft_objs = np.mean(mean_ft_objs, axis=0)    
    mean_random_objs = np.mean(mean_random_objs, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = list(range(1, len(mean_bc_objs)))
    plt.step(xs, mean_random_objs[1:], label='Random-LNS')
    plt.step(xs, mean_bc_objs[1:], label='BC-LNS')
    plt.step(xs, mean_ft_objs[1:], label='FT-LNS')
    ax.set_title('Objective Values with LNS Iterations')
    ax.set_xlabel('LNS Iteraions')
    ax.set_ylabel('Objective')
    ax.legend()
    plt.savefig('plots/psulu30step.png', bbox_inches='tight')
    

def aggregate_random_decomps_stats(logs_dir, plt_title, min_k, max_k, steps):
    aggregate_mean = []
    aggregate_stderr = []
    k_range = list(range(min_k, max_k+1))

    for k in k_range:
        aggregate_k = []
        output_k = os.path.join(logs_dir, 'random_k_%d_step_%d' %(k, steps))
        
        for log_f in os.listdir(output_k):
            abs_path = os.path.join(output_k, log_f)
            diff_data = parser.parse_random_decomp_log(abs_path)
            aggregate_k.extend(diff_data)

        aggregate_mean.append(np.mean(aggregate_k))
        aggregate_stderr.append(np.std(aggregate_k) / np.sqrt(len(aggregate_k)))

    plt.figure()
    plt.errorbar(k_range, aggregate_mean, yerr=aggregate_stderr, fmt='o-')
    plt.title('mean improvements vs cluster numbers')
    plt.show()
    print(aggregate_mean)
    print(aggregate_stderr)
    

def main(argv):
    logs_dir = FLAGS.logs_dir
    plt_title = FLAGS.plt_title
    min_k = FLAGS.min_k
    max_k = FLAGS.max_k
    steps = FLAGS.steps
    bc_dir = FLAGS.bc_dir
    ft_dir = FLAGS.ft_dir
    random_dir = FLAGS.random_dir
    gurobi_dir = FLAGS.gurobi_dir
    plt_name = FLAGS.plt_name
    mode = FLAGS.mode
    
    # plot_random_decomps_stats(logs_dir, plt_title)
    # aggregate_random_decomps_stats(logs_dir, plt_title, min_k, max_k, steps)

    if mode == 'step':
        plot_obj_step(bc_dir, ft_dir, random_dir)
    elif mode == 'time':
        plot_obj_time(bc_dir, ft_dir, random_dir, gurobi_dir, plt_name)
    elif mode == 'visualize':
        lp_file = 'data/psulu/20_20/lpfiles/valid/input7.lp'
        obst_file = 'data/psulu/20_20/obstacles/valid/envi7.yaml'
        best_cluster_f = 'random_logs/data/psulu/20_20/lpfiles/valid/random_k_3_step_10_time_2_init_0/best_clusters.pkl'
        visualize(lp_file, 20, 20, obst_file, 'plots/visualize', best_cluster_f, 5)

    
if __name__ == "__main__":
    app.run(main)
