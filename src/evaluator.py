import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import copy
from tqdm import tqdm as tqdm
import yaml
from collections import defaultdict
import argparse
import pickle
import os, shutil
import time

from envs.multigoalenv import createMultiGoalEnv
from src.symbolic_encoder import SymbolicEncoder
from src.unet_model import UNetXY
from src.q_function import QFunction
from src.parallelization import RLRunner
from src.train import *

class Evaluator(Trainer):
    def __init__(self,
                 device,
                 sym_encoder_kwargs,
                 q_function_kwargs,
                 env_configs,
                 n_envs,
                 reset_env_kwargs,
                 ep_length,
                 fdir=None,
                 detached_mode=True,
                 provide_subgoals=True,
                 use_structured_encodings=True,
                 **kwargs,
                ):
        # create envs, q function, and sym encoder
        self.detached_mode = detached_mode
        self.log_file_path = None
        self.device = device
        self.n_envs = n_envs
        self.reset_env_kwargs = reset_env_kwargs
        self.env_configs = env_configs
        self.envs = self.create_envs(env_configs)
        self.planner = SymbolicPlanner(env_configs['max_height'])
        self.time_to_reach_subgoal = 4

        self.use_structured_encodings = env_configs['use_structured_encodings']
        # this is needed for enc to tensor 
        self.max_enc_val = 6 if self.use_structured_encodings else self.planner.n_structures

        self.q_function = self.create_q_function(q_function_kwargs)
        self.sym_encoder = self.create_sym_encoder(sym_encoder_kwargs)
        self.ep_length = ep_length
        self.provide_subgoals = provide_subgoals

        # load models if files are provided
        if fdir is not None:
            self.q_function.load_network(fdir)
            # self.q_function.eval()
            # self.sym_encoder.load_network(fdir)
            # self.sym_encoder.eval()
            self.fdir = fdir

    def plot_trajectory(self, only_failure=False):
        assert self.n_envs == 1

        while True:
            traj_loaders = [TrajectoryLoader() for _ in range(self.n_envs)]
            obs, s_enc, g_enc = self.envs.reset()
            for i in range(self.n_envs):
                traj_loaders[i].goal = g_enc[i]
                traj_loaders[i].in_hands.append(obs[1][i])
                traj_loaders[i].top_downs.append(obs[2][i])
                traj_loaders[i].side_views.append(obs[3][i])
                traj_loaders[i].states.append(s_enc[i])

            active_envs = np.ones(self.n_envs, dtype=bool)
            print('running trajectories...')
            for step_id in range(self.ep_length):
                active_env_nums = np.where(active_envs)[0]
                state = self._state_to_tensor((*obs, s_enc, g_enc))

                a = self.q_function.action_selection(state)

                obs, s_enc, done = self.envs.step(a, active_env_nums)
                r = self.compute_reward(s_enc, g_enc, done, obs[0]).flatten()

                for i in range(len(active_env_nums)):
                    env_num = active_env_nums[i]
                    traj_loaders[env_num].goal = g_enc[i]
                    traj_loaders[env_num].rewards.append(r[i])
                    traj_loaders[env_num].in_hands.append(obs[1][i])
                    traj_loaders[env_num].top_downs.append(obs[2][i])
                    traj_loaders[env_num].side_views.append(obs[3][i])
                    traj_loaders[env_num].states.append(s_enc[i])
                    traj_loaders[env_num].q_maps.append(self.q_function.last_qmap[i])
                    traj_loaders[env_num].q_maps_noise.append(self.q_function.last_qmap_noise[i])
                    traj_loaders[env_num].actions.append(a[i])

                still_active_mask = np.bitwise_not(
                                        np.bitwise_or(done.flatten(), r.flatten()))
                active_envs[active_env_nums] = still_active_mask
                obs = [o[still_active_mask] for o in obs]
                s_enc = s_enc[still_active_mask]
                g_enc = g_enc[still_active_mask]
                if (active_envs==False).all():
                    break
            if r[0]==False:
                break
        print('plotting trajectories now')
        shutil.rmtree('trajectories')
        os.mkdir('trajectories')
        [t.save(i,'trajectories') for i,t in enumerate(traj_loaders)]

    def watch_in_pybullet(self, n_runs=100, dist_crit='local', build_mode='any',
                          state=None, goal=None):
        '''Creates rendered pb simulation and performs greedy policy on it
        '''
        def plot_it():
            def stack(structure_desc):
                layers = structure_desc.split(',')
                return '\n'.join(reversed(layers))
            f = plt.figure()
            gs = gridspec.GridSpec(2,2)
            ax0 = plt.subplot(gs[0,0])
            ax1 = plt.subplot(gs[0,1])
            ax2 = plt.subplot(gs[1,:])
            ax0.imshow(obs[2][0,...,0])
            ax0.axis('off')
            ax1.imshow(self.q_function.last_qmap[0])
            ax1.plot(a[0,2],a[0,1],'r.')
            ax1.axis('off')
            ax2.text(0.1,0.4,stack(env.enc_to_desc(s_enc)[0])+'\nstate',
                        color='red', fontweight='bold')
            ax2.text(0.5,0.4,stack(env.enc_to_desc(subg_enc)[0])+'\nsubgoal',
                        color='red', fontweight='bold')
            ax2.text(0.9,0.4,stack(env.enc_to_desc(g_enc)[0])+'\ngoal',
                        color='red', fontweight='bold')
            ax2.axis('off')
            plt.show()
            plt.close()

        reset_kwargs = {'dist_crit': dist_crit,
                        'build_mode': build_mode,
                        'state':state,
                        'goal':goal}
        configs = copy.copy(self.env_configs)
        configs['render'] = True
        env = RLRunner([createMultiGoalEnv(None, configs)])
        n_success = 0
        for _ in range(n_runs):
            obs, s_enc, g_enc = env.reset(**reset_kwargs)
            done = np.zeros_like(obs[0])
            opt_path_length = env.opt_path_length([0])[0]
            subg_enc = s_enc.copy()
        #     if s_enc[0,1] not in (1,2):
        #         continue
            # print(env.enc_to_desc(g_enc)[0])
            # print()

            for step_id in range(opt_path_length*4):
                r = self.compute_reward(s_enc, subg_enc, done, obs[0])
                if r[0]:
                    subg_enc = env.get_subgoal(s_enc)
                state = self._state_to_tensor((*obs, s_enc, subg_enc))
                # print(env.enc_to_desc(s_enc)[0])
                # print('?' if s_enc[0,-1] else '')
                a = self.q_function.action_selection(state)

                plot_it()
                obs, s_enc, done = env.step(a)

                if done[0]:
                    print('done')
                    break
                if self.compute_reward(s_enc, g_enc, done, obs[0])[0]:
                    n_success += 1
                    print('success')
                    break
        print(n_success/n_runs)

    def eval_q2(self, n_runs=100, use_pbar=True, show=True, **reset_kwargs):
        '''Runs 100 episodes

        :return:
        '''
        if use_pbar and not self.detached_mode:
            pbar = tqdm(total=n_runs)

        performance = [] # tuples (n_sg_reached, n_sg_total, n_actions_taken)
        active_envs = np.zeros(self.n_envs, dtype=bool)
        steps_left = np.zeros(self.n_envs, dtype=bool)
        n_subgoals_total = np.zeros(self.n_envs, dtype=np.uint8)
        n_subgoals_reached = np.zeros(self.n_envs, dtype=np.uint8)
        n_actions_taken = np.zeros(self.n_envs, dtype=np.uint8)
        n_actions_on_sg = np.zeros(self.n_envs, dtype=np.uint8)

        reset_batch_size = 2
        steps_per_sg = 6

        # initialize data
        obs, s_enc, g_enc = self.envs.reset()
        subg_enc = s_enc.copy()
        total_rewards = 0
        while len(performance) < n_runs:
            if (active_envs==0).sum() >= reset_batch_size:
                # reset environments that need to be reset
                env_nums_to_reset = np.where(active_envs==0)[0]
                _obs, _s_enc, _g_enc = self.envs.reset_envs(env_nums_to_reset,
                                                            **reset_kwargs)
                _opt_lengths = self.envs.opt_path_length(env_nums_to_reset)
                _subg_enc = self.envs.get_some_subgoals(env_nums_to_reset,
                                                       _s_enc)
                for i in range(len(obs)):
                    obs[i][active_envs==0] = _obs[i].copy()
                s_enc[active_envs==0] = _s_enc.copy()
                g_enc[active_envs==0] = _g_enc.copy()
                subg_enc[active_envs==0] = _subg_enc.copy()
                n_subgoals_reached[active_envs==0] = 0
                n_actions_taken[active_envs==0] = 0
                n_actions_on_sg[active_envs==0] = 0
                n_subgoals_total[active_envs==0] = _opt_lengths.copy()
                active_envs[:] = True

            active_env_nums = np.where(active_envs)[0]
            state = self._state_to_tensor((
                                *[o[active_envs] for o in obs],
                                s_enc[active_envs],
                                subg_enc[active_envs]))
            a = self.q_function.action_selection(state)
            n_actions_taken[active_env_nums] += 1
            n_actions_on_sg[active_env_nums] += 1

            new_obs, new_s_enc, new_done = self.envs.step(a, active_env_nums)
            done = np.zeros((self.n_envs, 1), dtype=bool)
            done[active_envs] = new_done
            s_enc[active_envs] = new_s_enc
            for i in range(len(obs)):
                obs[i][active_envs] = new_obs[i]

            # check for reaching subgoals
            subg_r = np.bitwise_and(self.compute_reward(s_enc, subg_enc, done, obs[0]).flatten(),
                                    active_envs)
            if subg_r.any():
                env_nums_that_reached_sg = np.where(subg_r)[0]
                n_subgoals_reached[env_nums_that_reached_sg] += 1
                subg_enc[subg_r] = self.envs.get_some_subgoals(
                                        env_nums_that_reached_sg,
                                        s_enc[subg_r]
                )
                n_actions_on_sg[env_nums_that_reached_sg] = 0

            # Check to see what envs must be reset
            r = self.compute_reward(s_enc, g_enc, done, obs[0]).flatten()
            out_of_steps = n_actions_on_sg > steps_per_sg

            terminal_envs = np.bitwise_and(r+out_of_steps+done.flatten() > 0,
                                            active_envs)
            active_envs[terminal_envs] = 0
            # push stats to performance array
            for env_num in np.where(terminal_envs)[0]:
                performance.append((n_subgoals_reached[env_num],
                                    n_subgoals_total[env_num],
                                    n_actions_taken[env_num]))

            if use_pbar and not self.detached_mode:
                pbar.update(terminal_envs.sum())

        performance = np.array(performance)

        max_length = performance[:,1].max()
        successes = np.zeros(max_length)
        counts = np.zeros(max_length)
        for i in range(max_length):
            successes[i] = (performance[:,0] > i).sum()
            print(f"length: {i}; n_successes: {(performance[:,0]==(i+1)).sum()}; n_attempts: {(performance[:,1]==(i+1)).sum()}")
            counts[i] = (performance[:,1] > i).sum()
        success_rates = successes/counts
        if show:
            plt.figure()
            plt.bar(np.arange(max_length)+1, success_rates)
            for i, p in enumerate(success_rates):
                va = 'bottom' if p < 0.95 else 'top'
                plt.text(i+1,p, f"{p:0.2f}", ha='center', va=va, fontweight='bold')
            plt.ylabel('success rate')
            plt.xlabel('symbolic distance')
            plt.ylim(0,1)
            plt.show()

            plt.figure()
            inc_success_rates = np.copy(success_rates)
            for i in range(1,len(success_rates)):
                inc_success_rates[i] = success_rates[i]/success_rates[i-1]
            plt.bar(np.arange(max_length)+1, inc_success_rates)
            for i, p in enumerate(inc_success_rates):
                va = 'bottom' if p < 0.95 else 'top'
                plt.text(i+1,p, f"{p:0.2f}", ha='center', va=va, fontweight='bold')
            plt.ylabel('success rate')
            plt.xlabel('symbolic distance')
            plt.ylim(0,1)
            plt.title('incremental')
            plt.show()

        return success_rates

    def compare_vs_opt_length(self, runs_per_datapoint=10, max_opt_length=5, show=False,
                              build_mode='any', state=None, goal=None, use_pbar=False):
        performances = []
        for i in range(1,max_opt_length+1):
            p = self.eval_q_function(n_runs=runs_per_datapoint, dist_crit=(i,i),
                                     build_mode=build_mode, state=state, goal=goal,
                                     use_pbar = use_pbar)
            # print(i, p)
            performances.append(p)
        if show:
            plt.figure()
            plt.bar(np.arange(1, max_opt_length+1), performances)
            for i, p in enumerate(performances):
                va = 'bottom' if p < 0.95 else 'top'
                plt.text(i+1,p, f"{p:0.2f}", ha='center', va=va, fontweight='bold')
            plt.xlabel('opt length')
            plt.ylabel('success rate')
            plt.ylim(0,1)
            plt.show()
        return performances

    def compare_along_path(self, start, goal, runs_per_datapoint=10, show=False,
                           use_pbar=False, incremental=False):
        assert start != goal, "start and goal must be different"
        performances = []
        path = self.envs.get_path(start, goal)[0]
        for i, goal in enumerate(path[1:]):
            state = path[i] if incremental else start
            performances.append(self.eval_q2(n_runs=runs_per_datapoint,
                                             state = state,
                                             goal = goal,
                                             use_pbar=use_pbar,
                                             show=False)[-1]
                               )
        if show:
            plt.figure()
            plt.bar(1+np.arange(len(performances)), performances)
            for i, p in enumerate(performances):
                va = 'bottom' if p < 0.95 else 'top'
                plt.text(i+1,p, f"{p:0.2f}", ha='center', va=va, fontweight='bold')
            plt.xlabel('opt length')
            plt.ylabel('success rate')
            readable_strings = []
            for s in path[1:]:
                readable_strings.append('\n'.join(s.split(',')[::-1]))
            plt.xticks(np.arange(1,len(path)), readable_strings)
            plt.title(f"{start} => {goal}")
            plt.ylim(0,1)
            plt.show()
        return performances

    def compare_build_modes(self, runs_per_datapoint=10, max_opt_length=5):
        build_modes = ['any', 'construct', 'deconstruct']
        colors = ['r','g','b']
        performances = dict()

        for bm in build_modes:
            print(bm)
            performances[bm] = self.compare_vs_opt_length(runs_per_datapoint,
                                                         max_opt_length,
                                                         build_mode=bm)

        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(111)

        rects = []
        ind = np.arange(max_opt_length)
        width = 0.27
        for i, bm in enumerate(build_modes):
            rects.append(ax.bar(ind+width*(i-1), performances[bm],
                                width=width, color=colors[i], alpha=0.6))
            for j in ind:
                va = 'top' if  performances[bm][j] > 0.1 else 'bottom'
                plt.text(j+width*(i-1), performances[bm][j], f"{performances[bm][j]:.2f}", ha='center', va=va)
        ax.set_ylim(0,1)
        ax.set_ylabel('Success Rate')
        ax.set_xlabel('Optimal Symbolic Path Length')
        ax.set_xticklabels([str(i) for i in np.arange(0,max_opt_length+1)])
        ax.legend(build_modes)
        plt.savefig(f'{self.fdir}/compare_build_modes.png')
        plt.close()

    def eval_sym_encoder(self, n_samples=100, n_steps=4):
        side_views = None
        encs = None

        n_total = 0
        if not self.detached_mode:
            pbar = tqdm(total=n_samples)
        while n_total < n_samples:
            obs, s_enc, g_enc = self.envs.reset()
            if side_views is None:
                side_views = obs[3]
                encs = s_enc
            else:
                side_views = np.concatenate((side_views, obs[3]),axis=0)
                encs = np.concatenate((encs, s_enc),axis=0)
            if len(side_views) > n_samples:
                break
            active_envs = np.ones(self.n_envs, dtype=bool)
            for _ in range(n_steps):
                active_env_nums = np.where(active_envs)[0]
                state = self._state_to_tensor((*obs, s_enc, g_enc))

                a = self.q_function.action_selection(state)

                obs, s_enc, done = self.envs.step(a, active_env_nums)

                active_envs[active_env_nums] = 1-done.flatten()
                obs = [o[done.flatten()==False] for o in obs]
                s_enc = s_enc[done.flatten()==False]
                g_enc = g_enc[done.flatten()==False]
                if active_envs.any():
                    side_views = np.concatenate((side_views, obs[3]),axis=0)
                    encs = np.concatenate((encs, s_enc),axis=0)
                else:
                    break
            if not self.detached_mode:
                pbar.update(len(side_views)-n_total)
            n_total = len(side_views)

        side_views = self._img_to_tensor(side_views)
        encs = self._enc_to_tensor(encs)
        pred_encs = self.sym_encoder.forward(side_views)
        breakdown = self.sym_encoder.accuracy_breakdown(pred_encs, encs)
        breakdown = breakdown.cpu().numpy()
        breakdown[:,:-1] /= breakdown[:,-1,None]

        max_height = encs.shape[1]
        row_labels = [f"height-{i+1}" for i in range(max_height)] + ['all']
        col_labels = [f"layer-{i+1} %" for i in range(max_height)] + ['total %', 'count']

        f = plt.figure(figsize=(6,2))
        cellText = [[f"{a:.1%}" if (i != max_height+1) else f"{int(a)}" for i,a in enumerate(row)] for row in breakdown]
        plt.table(cellText=cellText,
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      loc='center',
                      cellColours= max_height*[(max_height+2)*['w']]+[(max_height+2)*['#69bdd2']])
        plt.gca().axis('off')
        plt.show()

    def compare_per_goal(self, goals, start='___,___,___',
                         runs_per_datapoint=10, use_pbar=False, show=False):
        def readable(goal):
            return '\n'.join(reversed(goal.split(',')))

        success_rates = []
        max_len = 0
        pbar = tqdm(goals) if use_pbar else goals
        for goal in pbar:
            sr = self.eval_q2(n_runs=runs_per_datapoint,
                                              use_pbar=False,
                                              show=False,
                                              state=start,
                                              goal=goal)
            max_len = max(max_len, len(sr))
            success_rates.append(sr)

        tmp = np.zeros((len(goals),max_len))
        for i, s in enumerate(success_rates):
            tmp[i,(max_len-len(s)):] = s

        success_rates = tmp

        if show:
            plt.figure(figsize=(len(goals)*0.8,5))

            for i in range(max_len):
                bottom = None if i==0 else success_rates[:,i-1]
                values = success_rates[:,i] if i==0 else success_rates[:,i]-success_rates[:,i-1]
                plt.bar(np.arange(len(goals))+1, values, width=0.8,
                        color='#3d88c2', edgecolor='k', bottom=bottom)

            for i, s in enumerate(success_rates[:,-1]):
                plt.text(i+1,s, f"{s:0.2f}", ha='center',
                         va='top', fontweight='bold')

            plt.gca().set_xticks(np.arange(len(goals))+1)
            plt.gca().set_xticklabels([readable(g) for g in goals])
            plt.ylabel('success rate')
            plt.gca().set_axisbelow(True)
            plt.gca().yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
            plt.gca().yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.025))
            plt.grid(b=True, which='major', axis='y', color='k', ls='-', alpha=0.6)
            plt.grid(b=True, which='minor', axis='y', color='k', ls='--', alpha=0.2)
            plt.ylim(0.0,1)
            plt.show()
        return success_rates

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detached',action="store_true")
    args = parser.parse_args()
    if args.detached:
        print('--running detached--')


    fdir = f"{os.getcwd()}/results/26_h4_new_encoder_onehot"

    configs = load_training_configs(f"{fdir}/configs.yaml")
    configs['n_envs'] = 5
    configs['env_configs']['seed'] = np.random.randint(10,1000)

    configs['use_structured_encodings'] = False
    configs['env_configs']['use_structured_encodings'] = True

    # if 'env' in training_configs:
        # other_env_configs = training_configs.pop('env')
        # print(other_env_configs)
        # # overwrite any configs
        # env_configs = {**env_configs, **other_env_configs}
    # print(env_configs)

    # env_configs['use_structured_encodings'] = training_configs['use_structured_encodings']

    evaluator = Evaluator(
        device='cuda:0',
        fdir=fdir,
        detached_mode=args.detached,
        **configs
    )
    evaluator.eval_q_function(n_runs=200, use_pbar=True, show=True)
    # evaluator.compare_per_goal(goal,start=state,
                               # runs_per_datapoint=100,
                               # use_pbar=True, show=True)
    # exit()

    # evaluator.compare_build_modes(200, 5)
    # evaluator.compare_vs_opt_length(300, state='___,___,___',
                                    # build_mode='any', show=True, use_pbar=True)
    # evaluator.compare_vs_opt_length(state='___,___,___,___')
    # evaluator.plot_trajectory(only_failure=True)
    # print(evaluator.eval_q_function(n_runs=1000, state='___,___,___',
                                    # dist_crit='global'))
                                    # goal='c_c,_b_,c_c'))
    # goal = '_b_,_b_,c_c'
    # evaluator.watch_in_pybullet(**evaluator.reset_env_kwargs)

    # exit()
    # fig, ax = plt.subplots()
    # x = np.arange(len(perf1))
    # width = 0.45
    # rects1 = ax.bar(x-width/2, perf1, width=width, label='uniform-transition')
    # rects2 = ax.bar(x+width/2, perf2, width=width, label='uniform-goal')
    # for rects in [rects1, rects2]:
        # for rect in rects:
            # height = rect.get_height()
            # ax.annotate(f'{height:.2f}',
                        # xy=(rect.get_x() + rect.get_width() / 2, height),
                        # xytext=(0, -3),  # 3 points vertical offset
                        # textcoords="offset points", fontweight='bold',
                        # ha='center', va='top')
    # plt.xlabel('opt length')
    # plt.ylabel('success rate')
    # path = evaluator.envs.get_path(start='___,___,___', goal="c_c,_b_,c_c")[0]
    # readable_strings = []
    # for s in path[1:]:
        # readable_strings.append('\n'.join(s.split(',')[::-1]))
    # print(x)
    # print(readable_strings)
    # ax.set_xticks(x)
    # ax.set_xticklabels(readable_strings)
    # ax.legend(loc="lower right")
    # # plt.title(f"{path[0]} => {path[-1]}")
    # plt.ylim(0,1)
    # plt.show()
    print(f"env seed = {env_configs['seed']}")

    # data = np.load(f"{fdir}/q_function_loss.npy")
    # plt.figure()
    # plt.plot(*data)
    # data = np.load(f"{fdir}/rewards.npy")
    # window = 100
    # rewards = np.convolve(data[1], np.ones(window)/window, 'valid')
    # plt.figure()
    # plt.plot(rewards)
    # plt.show()
