import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import copy
from tqdm import tqdm as tqdm
import argparse
import os
from collections import defaultdict
import time

from envs.multigoalenv import createMultiGoalEnv
from envs.goal_space import StructuredGoalSpace
from src.symbolic_encoder import SymbolicEncoder
from src.unet_model import UNetXY
from src.model_utils import rand_perlin_2d
from src.q_function import QFunction
from src.memory import create_memory
from src.parallelization import RLRunner
from configs import get_env_configs, get_training_configs

PARAMS = {
    'Ours' : {'goal_label_method' : 'subgoal',
              'use_local_policy' : True,
              'env_reset_goal_distribution' : 'terminal',
             },
    'HER'  : {'goal_label_method' : 'her',
              'use_local_policy' : False,
              'env_reset_goal_distribution' : 'all',
             },
    'UVFA' : {'goal_label_method' : 'finalgoal',
              'use_local_policy' : False,
              'env_reset_goal_distribution' : 'all',
             },
    'Shaped' : {'goal_label_method' : 'stepwise',
                  'reward_style' : 'stepwise',
                  'gamma' : 0.9,
                  'use_local_policy' : False,
                  'env_reset_goal_distribution' : 'terminal',
                 },
    'NeighborReplay' : {'goal_label_method' : 'neighbor',
                         'use_local_policy' : False,
                         'env_reset_goal_distribution' : 'terminal',
                        },
    'Curriculum' : {'goal_label_method' : 'finalgoal',
                    'use_local_policy' : False,
                   },
}

class EpisodeData:
    def __init__(self, n_envs):
        self.active_envs = np.zeros(n_envs, dtype=bool)
        self.n_actions_on_sg = np.zeros(n_envs, dtype=int)
        self.n_actions_taken = np.zeros(n_envs, dtype=int)
        self.n_subgoals_reached = np.zeros(n_envs, dtype=int)
        self.n_subgoals_total = np.zeros(n_envs, dtype=int)
        self.is_terminal = np.zeros(n_envs, dtype=int)

def add_perlin(obs, c, name):
    cycles = {'top_down' : [1,2,3,5,6],
              'in_hand' : [1,2],
              'side_view' : [1,2,3,4],
             }[name]

    N, C, H, W = obs.shape
    img_shape = (H, W)
    for i in range(obs.size(0)):
        res = np.random.choice(cycles, 2)
        noise = c * rand_perlin_2d(img_shape, res).to(obs.device)
        obs[i, 0] += noise

    return obs

def gaussian_smooth(obs, k_size=7):
    mean = k_size//2
    sigma = np.random.uniform(0.5,1)
    ygrid, xgrid = torch.meshgrid([torch.arange(k_size)-mean, torch.arange(k_size)-mean])
    kernel = np.exp( - (xgrid*xgrid + ygrid*ygrid) / (2*sigma**2) )
    kernel /= kernel.sum()
    kernel = kernel.view(1,1,k_size,k_size).to(obs.device)

    inp = torch.nn.functional.pad(obs, 4*[mean], mode='reflect')
    return torch.nn.functional.conv2d(inp, kernel)

class Trainer():
    def __init__(self,
                 fdir,
                 env_configs,
                 device,
                 sym_encoder_lr,
                 buffer_size,
                 goal_label_method,
                 q_opt_cycles,
                 enc_opt_cycles,
                 sym_encoder_img_size,
                 batch_size,
                 gamma,
                 reward_style,
                 n_envs,
                 env_reset_goal_distribution,
                 time_to_reach_subgoal,
                 perlin_noise=0,
                 add_smoothing=0,
                 specific_goal=None,
                 use_local_policy=True,
                 policy_locality=1,
                 detached_mode=False,
                 q_target_update_freq=1000,
                 include_s_enc=True,
                 **kwargs,
                ):
        self.fdir = fdir
        self.detached_mode = detached_mode
        self.log_file_path = self.fdir+"/train.log"
        self.device = torch.device(device)
        self.max_height = env_configs['max_height']
        self.time_to_reach_subgoal = time_to_reach_subgoal
        self.ep_length = (time_to_reach_subgoal+1)*self.max_height

        self.q_opt_cycles = q_opt_cycles
        self.enc_opt_cycles = enc_opt_cycles
        self.batch_size = batch_size
        self.n_envs = n_envs
        self.env_reset_goal_distribution = env_reset_goal_distribution
        self.specific_goal = specific_goal
        self.use_local_policy = use_local_policy
        self.policy_locality = policy_locality

        self.perlin_noise = perlin_noise
        self.add_smoothing = add_smoothing

        self.envs = self.create_envs(env_configs, specific_goal)
        self.g_space = StructuredGoalSpace(env_configs['max_height'])
        if specific_goal is not None:
            self.g_space.constrain_to(specific_goal)

        if 't0' in env_configs['other_obj_chars']:
            self.g_space.add_triangle_object(id_=0)
        if 't1' in env_configs['other_obj_chars']:
            self.g_space.add_triangle_object(id_=1)
        if 'y' in env_configs['other_obj_chars']:
            self.g_space.add_cylinder_object()

        self.use_structured_encodings = env_configs['use_structured_encodings']

        # this is needed for enc to tensor 
        self.max_enc_val = 6 if self.use_structured_encodings else len(self.g_space.all_structures)

        self.q_function = self.create_q_function(gamma, reward_style,
                                                 q_target_update_freq, include_s_enc)
        self.sym_encoder = self.create_sym_encoder(sym_encoder_img_size,
                                                   sym_encoder_lr)

        self.memory = self.create_memory(self.ep_length, buffer_size, goal_label_method)

    def create_envs(self, env_config, specific_goal=None):
        '''
        Load a list of worker envs
        '''
        env_create_fn = createMultiGoalEnv
        env_fns = []
        for i in range(self.n_envs):
            tmp_config = copy.copy(env_config)
            tmp_config['seed'] = env_config['seed']+i
            env_fns.append(env_create_fn(None, tmp_config, specific_goal))
        envs = RLRunner(env_fns)
        return envs

    def create_q_function(self, gamma, reward_style, target_update_freq, include_s_enc):
        obs, s_enc, _ = self.envs.reset()
        in_hand_shape = self._img_to_tensor(obs[1])[0].shape
        top_down_shape = self._img_to_tensor(obs[2])[0].shape
        enc_shape = self._enc_to_tensor(s_enc)[0].shape
        # load default kw_args
        return QFunction(in_hand_shape,
                         top_down_shape,
                         enc_shape,
                         gamma=gamma,
                         reward_style=reward_style,
                         target_update_freq=target_update_freq,
                         log_fn=self._log,
                         include_s_enc=include_s_enc,
                        ).to(self.device)

    def create_sym_encoder(self, img_size, lr):
        return SymbolicEncoder(img_size, self.max_enc_val,
                               lr=lr).to(self.device)

    def create_memory(self, ep_length, buffer_size, goal_label_method):
        obs, s_enc, _ = self.envs.reset()
        action = self.envs.random_action()
        g_space = StructuredGoalSpace(self.max_height)
        return create_memory(ep_length, obs, s_enc, action,
                             self.compute_reward,
                             buffer_size,
                             label_method=goal_label_method,
                             goal_space=g_space,
                            )

    def reset_as_needed(self,
                        ep_data,
                        transition,
                        env_reset_goal_distribution=None,
                        goal_subset=None,
                       ):
        if env_reset_goal_distribution is None:
            env_reset_goal_distribution = self.env_reset_goal_distribution
        if goal_subset is None and self.specific_goal is not None:
            goal_subset = [self.specific_goal]

        mask = ep_data.active_envs==0
        obs, s_enc, subg_enc, g_enc = transition
        if mask.any():
            # reset environments that need to be reset
            env_nums_to_reset = np.where(mask)[0]
            _obs, _s_enc, _g_enc = self.envs.reset_envs(env_nums_to_reset,
                                                        env_reset_goal_distribution,
                                                        goal_subset)
            _path_length = self.envs.opt_path_length(env_nums_to_reset)
            _subg_enc = self.envs.get_some_subgoals(env_nums_to_reset, _s_enc)
            self.memory.start_episodes(env_nums_to_reset, _obs,
                                       _s_enc, _g_enc)
            for i in range(len(obs)):
                obs[i][env_nums_to_reset] = _obs[i].copy()
            s_enc[env_nums_to_reset] = _s_enc.copy()
            g_enc[env_nums_to_reset] = _g_enc.copy()
            subg_enc[env_nums_to_reset] = _subg_enc.copy()

            ep_data.n_subgoals_reached[env_nums_to_reset] = 0
            ep_data.n_subgoals_total[env_nums_to_reset] = _path_length.copy()
            ep_data.n_actions_on_sg[env_nums_to_reset] = 0
            ep_data.n_actions_taken[env_nums_to_reset] = 0
            ep_data.is_terminal[env_nums_to_reset] = [self.envs.is_terminal_goal(i) for i in env_nums_to_reset]
            ep_data.active_envs[:] = True

        return ep_data, (obs, s_enc, subg_enc, g_enc)

    def set_new_subgoals(self, ep_data, subg_r, s_enc, subg_enc, r):
        mask = subg_r.flatten() == 1
        if mask.any():
            env_nums = np.where(mask)[0]
            ep_data.n_subgoals_reached[env_nums] += 1
            for env_num in env_nums:
                if r[env_num] == 0:
                    ep_data.n_actions_on_sg[env_nums] = 0

            update_mask = np.bitwise_and(mask, r == 0)
            if update_mask.any():
                update_env_nums = np.where(update_mask)[0]
                subg_enc[update_env_nums] = self.envs.get_some_subgoals(update_env_nums,
                                                  s_enc[update_env_nums])
        return ep_data, subg_enc

    def get_action(self, full_state, epsilon):
        obs, s_enc, subg_enc, g_enc = full_state

        if self.use_local_policy:
            state_tensor = self._state_to_tensor((*obs, s_enc, subg_enc))
        else:
            state_tensor = self._state_to_tensor((*obs, s_enc, g_enc))

        return self.q_function.action_selection(state_tensor, epsilon)

    def check_for_envs_to_reset(self, ep_data, r, done,
                                time_to_reach_subgoal=None):
        if time_to_reach_subgoal is None:
            time_to_reach_subgoal = self.time_to_reach_subgoal
            ep_length = self.ep_length
        else:
            ep_length = ep_data.n_subgoals_total * time_to_reach_subgoal

        out_of_steps = np.bitwise_or(
            ep_data.n_actions_on_sg >= time_to_reach_subgoal,
            ep_data.n_actions_taken >= ep_length
        )

        # terminal_envs = (r.flatten()+out_of_steps+done.flatten()) > 0
        terminal_envs = (out_of_steps+done.flatten()) > 0
        ep_data.active_envs[terminal_envs] = 0

        return ep_data

    def run_and_yield(self,
                      env_step_interval,
                      total_env_steps,
                      eps_range=(1.0, 0.0),
                      goal_subset=None,
                      novelty_eps=False,
                      optimal_period=0,
                      log_freq=1000,
                      step_count=0,
                      rewards_data=[]):
        episode_count = 0
        log_mod_counter = 0
        step_mod_counter = 0
        self.total_env_steps = total_env_steps

        #initialize data
        ep_data = EpisodeData(self.n_envs)
        obs, s_enc, g_enc = self.envs.reset(goal_distribution=self.env_reset_goal_distribution,
                                            goal_subset=goal_subset)
        subg_enc = self.envs.get_subgoals(s_enc)

        pbar = tqdm(total=total_env_steps)
        while step_count <= total_env_steps:
            pbar.update(1)
            full_state = (obs, s_enc, subg_enc, g_enc)
            ep_data, full_state = self.reset_as_needed(ep_data, full_state,
                                                      goal_subset=goal_subset)

            step_progress = min(step_count/(total_env_steps-optimal_period), 1)
            epsilon = eps_range[0]*(1-step_progress) + eps_range[1]*step_progress
            if novelty_eps:
                goals = [self.g_space.all_structures[enc[0]] for enc in full_state[2]]
                is_novel = [g in self.g_space.novel_structures for g in goals]
                epsilon = np.full(self.n_envs, epsilon)
                epsilon[np.bitwise_not(is_novel)] = 0.0

            a = self.get_action(full_state, epsilon)

            # for speed, we step env async and optimize while it runs
            self.envs.stepAsync(a)
            [self.optimize_models('q_function') for _ in range(self.q_opt_cycles)]

            obs, s_enc, subg_enc, g_enc = full_state
            obs, s_enc, done = self.envs.stepWait()
            ep_data.n_actions_on_sg += 1
            ep_data.n_actions_taken += 1
            step_count += self.n_envs
            log_mod_counter += self.n_envs
            step_mod_counter += self.n_envs

            # calculate rewards
            subg_r = self.compute_reward(s_enc, subg_enc, done, obs[0])
            r = self.compute_reward(s_enc, g_enc, done, obs[0])

            self.memory.add_to_episodes(np.arange(self.n_envs), obs, s_enc,
                                        subg_enc, a, subg_r, done,
                                        ep_data.n_subgoals_reached[:,None]+subg_r)

            # change subgoals as needed
            ep_data, subg_enc = self.set_new_subgoals(ep_data, subg_r,
                                                      s_enc, subg_enc, r)

            # determine what envs should be reset
            ep_data = self.check_for_envs_to_reset(ep_data, r, done)

            terminal_env_nums = np.where(ep_data.active_envs == 0)[0]
            self.memory.end_episodes(terminal_env_nums)
            episode_count += len(terminal_env_nums)

            rewards_data.extend([(step_count,
                                  ep_data.n_actions_taken[i],
                                  ep_data.n_subgoals_reached[i],
                                  ep_data.n_subgoals_total[i],
                                  ep_data.is_terminal[i],
                                 ) for i in terminal_env_nums])

            if log_mod_counter > log_freq:
                log_mod_counter = log_mod_counter % log_freq
                self.q_function.log_progress(self.fdir)
                self.sym_encoder.log_progress(self.fdir)
                np.save(f"{self.fdir}/rewards.npy", np.array(rewards_data).T)

            if step_mod_counter >= env_step_interval:
                step_mod_counter = step_mod_counter % env_step_interval
                yield step_count

            [self.optimize_models('encoder') for _ in range(self.enc_opt_cycles)]
        np.save(f"{self.fdir}/rewards.npy", np.array(rewards_data).T)
        yield step_count

    def optimize_models(self, models='both'):
        assert models in ('q_function', 'encoder', 'both')
        batch = self.memory.sample(self.batch_size)
        if batch is None:
            return
        else:
            (s,a,sp,r,d) = batch
        if models in ('q_function', 'both'):
            # r = self.compute_reward(sp[4], sp[5], d, sp[0])
            self.q_function.optimize((self._state_to_tensor(s),
                                      self._action_to_tensor(a),
                                      self._reward_to_tensor(r),
                                      self._state_to_tensor(sp),
                                      self._reward_to_tensor(d),
                                     ))
        if models in ('encoder', 'both'):
            self.sym_encoder.optimize(self._img_to_tensor(s[3]),
                                      self._enc_to_tensor(s[4]))
            self.sym_encoder.optimize(self._img_to_tensor(sp[3]),
                                      self._enc_to_tensor(sp[4]))

    def compute_reward(self, achieved_enc, g_enc, done, is_holding):
        return np.bitwise_and.reduce((
                    (achieved_enc == g_enc).all(axis=1, keepdims=True),
                    np.bitwise_not(done),
                    np.bitwise_not(is_holding)), axis=0)

    def _obs_to_tensor(self, obs):
        """
        Convert each element of the obs to tensor to be fed to network

        parameters:
            obs = (is_holding, : np array (N, 1), dtype=bool
                   in_hand,    : np array (N, H,W,C), dtype=int
                   top_down,   : np array (N, H,W,C), dtype=int
                   side_view,  : np array (N, H,W,C), dtype=int
                   )
        returns:
            t_obs = (is_holding, : tensor (N, 1), dtype=long
                     in_hand,    : tensor (N,C,H,W), dtype=float
                     top_down,   : tensor (N,C,H,W), dtype=float
                     side_view,  : tensor (N,C,H,W), dtype=float
                     )
        """
        t_obs = [
            torch.tensor(obs[0],dtype=torch.long,device=self.device),
            self._img_to_tensor(obs[1]),
            self._img_to_tensor(obs[2]),
            self._img_to_tensor(obs[3]),
        ]

        if self.perlin_noise > 1e-4:
            t_obs[1] = add_perlin(t_obs[1], c=self.perlin_noise, name='in_hand')
            t_obs[2] = add_perlin(t_obs[2], c=self.perlin_noise, name='top_down')
            t_obs[3] = add_perlin(t_obs[3], c=self.perlin_noise, name='side_view')
        if self.add_smoothing:
            t_obs[1] = gaussian_smooth(t_obs[1])
            t_obs[2] = gaussian_smooth(t_obs[2])
            t_obs[3] = gaussian_smooth(t_obs[3])

        return t_obs

    def _action_to_tensor(self, action):
        return torch.tensor(action, dtype=torch.long, device=self.device)

    def _reward_to_tensor(self, reward):
        return torch.tensor(reward, dtype=torch.float32, device=self.device)

    def _img_to_tensor(self, img):
        t = torch.tensor(img,dtype=torch.float32,device=self.device)
        return t.permute(0,3,1,2)/255.

    def _enc_to_tensor(self, enc):
        """
        Convert symbolic encoding to one-hot tensor

        :inputs:
            :enc: list of numpy arrays where the i-th element is a categorical
                 description of what exists in the i-th level of the structure
        :returns:
            :t_enc: tensor where the categorical descriptions for each level are
               expanded to one-hot encodings
        """
        if self.use_structured_encodings:
            t_enc = torch.zeros((enc.shape[0]*enc.shape[1], self.max_enc_val),
                                device=self.device,dtype=torch.float32)
            idx = torch.tensor(enc, dtype=torch.long, device=self.device) \
                                .flatten().unsqueeze(1)
            t_enc = t_enc.scatter_(1, idx, 1.)
            t_enc = t_enc.view(enc.shape[0], enc.shape[1],-1)
        else:
            t_enc = torch.zeros((enc.shape[0], self.max_enc_val+1),
                                device=self.device,dtype=torch.float32)
            # t_enc[np.arange(enc.shape[0]),enc[:,0]] = 1
            idx = torch.tensor(enc[:,0], dtype=torch.long, device=self.device).unsqueeze(1)
            t_enc.scatter_(1, idx, 1.)
            t_enc[enc[:,1]==1,-1] = 1
        return t_enc

    def _log(self, string):
        # send to log file
        if self.log_file_path is not None:
            with open(self.log_file_path, 'a') as f:
                f.write(string+'\n')

    def _state_to_tensor(self, state):
        """Convert all elements in state to tensor.

        :input:
            :state: list of (is_holding, in_hand, top_down,
                                side_view, s_enc, g_enc)
        :return:
            :t_state: same as state but all elements are tensors
        """
        return (*self._obs_to_tensor(state[:4]), self._enc_to_tensor(state[4]),
                self._enc_to_tensor(state[5]))

    def eval_q_function(self, reset_goal_distribution='all',
                        goal_subset=None, n_runs=100,
                        time_to_reach_subgoal=4):
        eval_data = {'terminal' : [0,0],
                     'all' : [0,0]}
        progress_data = []

        # initialize data
        ep_data = EpisodeData(self.n_envs)
        obs, s_enc, g_enc = self.envs.reset(reset_goal_distribution,
                                            goal_subset)
        subg_enc = self.envs.get_subgoals(s_enc)

        total_rewards = 0
        t = time.time()
        while len(progress_data) < n_runs:
            full_state = (obs, s_enc, subg_enc, g_enc)
            ep_data, full_state = self.reset_as_needed(ep_data, full_state,
                                                       reset_goal_distribution,
                                                       goal_subset,
                                                      )

            # pick action with no exploration noise added
            a = self.get_action(full_state, epsilon=0.0)

            obs, s_enc, done = self.envs.step(a)
            ep_data.n_actions_on_sg += 1
            ep_data.n_actions_taken += 1

            # calculate rewards
            subg_r = self.compute_reward(s_enc, subg_enc, done, obs[0])
            r = self.compute_reward(s_enc, g_enc, done, obs[0])

            # change subgoals as needed
            ep_data, subg_enc = self.set_new_subgoals(ep_data, subg_r,
                                                      s_enc, subg_enc, r)

            # ids = np.where(r==1)[0]
            # for i in ids:
                # self.envs.remotes[i].send(('get_obj_states', None))
                # states = self.envs.remotes[i].recv()
                # return states

            # Check to see what envs must be reset
            ep_data = self.check_for_envs_to_reset(ep_data, r, done,
                                                   time_to_reach_subgoal)

            # push stats to performance array
            for env_num in np.where(ep_data.active_envs==0)[0]:
                progress_data.append((ep_data.n_subgoals_reached[env_num],
                                      ep_data.n_subgoals_total[env_num]))

                eval_data['all'][0] += r[env_num][0]
                eval_data['all'][1] += 1
                if self.envs.is_terminal_goal(env_num):
                    eval_data['terminal'][0] += r[env_num][0]
                    eval_data['terminal'][1] += 1

        return eval_data, np.array(progress_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='./results')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_height', type=int, default=1)
    parser.add_argument('--num_env_steps', type=int, default=50_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', type=str, default='Ours',
                        choices=PARAMS.keys())
    args = parser.parse_args()

    training_configs = get_training_configs(**PARAMS[args.method])
    env_configs = get_env_configs(max_height=args.max_height,
                                          seed=args.seed)

    if not os.path.exists(args.folder):
        os.mkdir(args.folder)

    trial_name = f'{args.method}_height{args.max_height}_seed{args.seed}'
    fdir = os.path.join(args.folder, trial_name)
    if not os.path.exists(fdir):
        os.mkdir(fdir)

    trainer = Trainer(fdir=fdir,
                      device=args.device,
                      env_configs=env_configs,
                      **training_configs,
                     )
    trainer._log(f"=== STARTING trial:'{trial_name}' on device:'{args.device}'" \
          f" @{time.strftime('%b %d %H:%M:%S', time.localtime())} ===")

    step_interval = int(args.num_env_steps/5)
    all_eval_data = {}
    all_progress_data = {}
    for step_count in trainer.run_and_yield(step_interval, args.num_env_steps+1):
        trainer.q_function.log_progress(trainer.fdir, f"{step_count}steps")

        # evaluate
        eval_data, progress_data = trainer.eval_q_function(reset_goal_distribution='all',
                                                          n_runs=1000)
        all_eval_data[str(step_count)] = eval_data
        all_progress_data[str(step_count)] = progress_data

        # save on the go
        np.savez(fdir+"/eval_data.npz", **all_eval_data)
        np.savez(fdir+"/progress_data.npz", **all_progress_data)

    # save configs
    with open(fdir+'/configs.yaml', 'w+') as f:
        all_configs = {'env_configs' : env_configs,
                       'training_configs' : training_configs}
        yaml.dump(all_configs, f)

    trainer._log(f"=== FINISHED trial:'{trial_name}' on device:'{args.device}'" \
          f" @{time.strftime('%b %d %H:%M:%S', time.localtime())} ===")
