import numpy as np
import time
import os

class AbstractMemory:
    def __init__(self,
                 ep_length,
                 enc,
                 size,
                ):
        self.current_size = 0
        self.size = size
        self.ep_length = ep_length
        self.n_envs = len(enc)
        self.initialize_data(enc)
        self.ep_batch = self.empty_ep_batch()

    def initialize_data(self, enc):
        self.data = {
            's_enc'     : np.zeros((self.size, self.ep_length+1,
                                 *enc[0].shape), dtype=enc[0].dtype),
            'sg_enc'     : np.zeros((self.size, self.ep_length,
                                 *enc[0].shape), dtype=enc[0].dtype),
            'g_enc'     : np.zeros((self.size, 1,
                                 *enc[0].shape), dtype=enc[0].dtype),
            'reward'  : np.zeros((self.size, self.ep_length, 1),dtype=int),
            'n_steps' : np.zeros((self.size), dtype=np.uint8),
        }

    def empty_ep_batch(self):
        empty_batch = {}
        for k in self.data.keys():
            shape = list(self.data[k].shape)
            shape[0] = self.n_envs
            empty_batch[k] = np.zeros(shape, self.data[k].dtype)
        return empty_batch

    def start_episodes(self, env_nums, s_enc, g_enc):
        self.ep_batch['s_enc'][env_nums,0] = s_enc.copy()
        self.ep_batch['g_enc'][env_nums,0] = g_enc.copy()

    def add_to_episodes(self, env_nums, s_enc, sg_enc, reward):
        self.ep_batch['n_steps'][env_nums] += 1
        step_ids = self.ep_batch['n_steps'][env_nums]
        self.ep_batch['s_enc'][env_nums,step_ids] = s_enc.copy()
        self.ep_batch['sg_enc'][env_nums,step_ids-1] = sg_enc.copy()
        self.ep_batch['reward'][env_nums,step_ids-1] = reward.copy()

    def end_episodes(self, env_nums, s_enc, sg_enc, reward):
        if len(env_nums) == 0:
            return
        self.add_to_episodes(env_nums, s_enc, sg_enc, reward)
        n_batches = len(env_nums)

        idxs = self._get_storage_idx(n_batches)

        for k in self.data.keys():
            self.data[k][idxs] = self.ep_batch[k][env_nums].copy()
            self.ep_batch[k][env_nums] = 0

        return idxs

    def get_sample_idxs(self, batch_size):
        ep_lengths = self.data['n_steps'][:self.current_size]
        ep_idxs = np.random.choice(np.arange(self.current_size),
                                   size=batch_size,
                                   p=ep_lengths/ep_lengths.sum())
        ep_idxs = np.random.randint(0, self.current_size, size=batch_size)
        ep_lengths = self.data['n_steps'][ep_idxs]
        t_idxs = np.random.randint(0, ep_lengths)
        return ep_idxs, t_idxs

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        # if inc == 1:
            # idx = idx[0]
        return idx

    def sample(self, batch_size):
        if len(self) < 10*batch_size:
            return None

        ep_idxs, t_idxs = self.get_sample_idxs(batch_size)
        s = self.data['s_enc'][ep_idxs, t_idxs]
        sg = self.data['sg_enc'][ep_idxs, t_idxs]
        sp = self.data['s_enc'][ep_idxs, t_idxs+1]
        g = self.data['g_enc'][ep_idxs, 0]
        r = self.data['reward'][ep_idxs, t_idxs]

        return s, sg, sp, g, r

    def __len__(self):
        return self.current_size*self.ep_length

class BaseMemory:
    def __init__(self,
                 ep_length,
                 obs,
                 enc,
                 action,
                 reward_function,
                 size,
                 **kwargs,
                ):
        self.current_size = 0
        self.size = size
        self.ep_length = ep_length
        self.n_envs = len(obs[0])
        self.reward_function = reward_function
        self.initialize_data(obs, enc, action)
        self.ep_batch = self.empty_ep_batch()

    def initialize_data(self,
                        obs,
                        enc,
                        action,
                       ):
        is_holding, in_hand, top_down, side_view = obs
        self.data = {
            'is_holding' : np.zeros((self.size, self.ep_length+1,
                             *is_holding[0].shape), dtype=is_holding[0].dtype),
            'in_hand'   : np.zeros((self.size, self.ep_length+1,
                             *in_hand[0].shape), dtype=in_hand[0].dtype),
            'top_down'  : np.zeros((self.size, self.ep_length+1,
                             *top_down[0].shape), dtype=top_down[0].dtype),
            'side_view'  : np.zeros((self.size, self.ep_length+1,
                                 *side_view[0].shape), dtype=side_view[0].dtype),
            's_enc'     : np.zeros((self.size, self.ep_length+1,
                                 *enc[0].shape), dtype=enc[0].dtype),
            'sg_enc'     : np.zeros((self.size, self.ep_length,
                                 *enc[0].shape), dtype=enc[0].dtype),
            'g_enc'     : np.zeros((self.size, 1,
                                 *enc[0].shape), dtype=enc[0].dtype),
            'actions'   : np.zeros((self.size, self.ep_length,
                                    *action[0].shape),dtype=action.dtype),
            'n_steps' : np.zeros((self.size), dtype=np.uint8),
            'dones'   : np.zeros((self.size, self.ep_length, 1),dtype=bool),
            'sg_rewards'  : np.zeros((self.size, self.ep_length, 1),dtype=int),
            'sg_reached' : np.zeros((self.size, self.ep_length, 1),dtype=int),
        }

    def empty_ep_batch(self):
        empty_batch = {}
        for k in self.data.keys():
            shape = list(self.data[k].shape)
            shape[0] = self.n_envs
            empty_batch[k] = np.zeros(shape, self.data[k].dtype)
        return empty_batch

    def start_episodes(self, env_nums, obs, s_enc, g_enc):
        self.ep_batch['is_holding'][env_nums,0] = obs[0].copy()
        self.ep_batch['in_hand'][env_nums,0] = obs[1].copy()
        self.ep_batch['top_down'][env_nums,0] = obs[2].copy()
        self.ep_batch['side_view'][env_nums,0] = obs[3].copy()
        self.ep_batch['s_enc'][env_nums,0] = s_enc.copy()
        self.ep_batch['g_enc'][env_nums,0] = g_enc.copy()

    def add_to_episodes(self, env_nums, obs, s_enc, sg_enc, a, sg_reward, done, sg_reached):
        self.ep_batch['n_steps'][env_nums] += 1
        step_ids = self.ep_batch['n_steps'][env_nums]
        try:
            self.ep_batch['is_holding'][env_nums,step_ids] = obs[0].copy()
            self.ep_batch['in_hand'][env_nums,step_ids] = obs[1].copy()
            self.ep_batch['top_down'][env_nums,step_ids] = obs[2].copy()
            self.ep_batch['side_view'][env_nums,step_ids] = obs[3].copy()
            self.ep_batch['s_enc'][env_nums,step_ids] = s_enc.copy()
            self.ep_batch['sg_enc'][env_nums,step_ids-1] = sg_enc.copy()
            self.ep_batch['actions'][env_nums,step_ids-1] = a.copy()
            self.ep_batch['dones'][env_nums,step_ids-1] = done.copy()
            self.ep_batch['sg_rewards'][env_nums,step_ids-1] = sg_reward.copy()
            self.ep_batch['sg_reached'][env_nums,step_ids-1] = sg_reached.copy()
        except IndexError as e:
            print('env_nums', env_nums)
            print('step_ids', step_ids)
            print('n_envs', self.n_envs)
            raise e

    def end_episodes(self, env_nums):
        if len(env_nums) == 0:
            return
        n_batches = len(env_nums)

        idxs = self._get_storage_idx(n_batches)

        for k in self.data.keys():
            self.data[k][idxs] = self.ep_batch[k][env_nums].copy()
            self.ep_batch[k][env_nums] = 0
        return idxs

    def get_sample_idxs(self, batch_size):
        ep_lengths = self.data['n_steps'][:self.current_size]
        ep_idxs = np.random.choice(np.arange(self.current_size),
                                   size=batch_size,
                                   p=ep_lengths/ep_lengths.sum())
        ep_idxs = np.random.randint(0, self.current_size, size=batch_size)
        ep_lengths = self.data['n_steps'][ep_idxs]
        t_idxs = np.random.randint(0, ep_lengths)
        return ep_idxs, t_idxs

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        # if inc == 1:
            # idx = idx[0]
        return idx

    def save(self, fdir):
        filename = os.path.join(fdir, 'memory.npz')
        np.savez(filename,
                 current_size=np.array(self.current_size),
                 **self.data)

    def clear(self):
        self.current_size = 0
        for k in self.data.keys():
            self.data[k][:] = 0

    def __len__(self):
        return self.current_size*self.ep_length

class LocalReplayMemory(BaseMemory):
    def sample(self, batch_size):
        if len(self) < 10*batch_size:
            return None

        ep_idxs, t_idxs = self.get_sample_idxs(batch_size)
        s = [
            self.data['is_holding'][ep_idxs, t_idxs],
            self.data['in_hand'][ep_idxs, t_idxs],
            self.data['top_down'][ep_idxs, t_idxs],
            self.data['side_view'][ep_idxs, t_idxs],
            self.data['s_enc'][ep_idxs, t_idxs],
            self.data['sg_enc'][ep_idxs, t_idxs]
        ]
        a = self.data['actions'][ep_idxs, t_idxs]
        sp = [
            self.data['is_holding'][ep_idxs, t_idxs+1],
            self.data['in_hand'][ep_idxs, t_idxs+1],
            self.data['top_down'][ep_idxs, t_idxs+1],
            self.data['side_view'][ep_idxs, t_idxs+1],
            self.data['s_enc'][ep_idxs, t_idxs+1],
            self.data['sg_enc'][ep_idxs, t_idxs]
        ]
        rewards = self.data['sg_rewards'][ep_idxs, t_idxs]
        dones = self.data['dones'][ep_idxs, t_idxs]

        return s, a, sp, rewards, dones

class MetaMemory(BaseMemory):
    def sample(self, batch_size):
        if len(self) < 10*batch_size:
            return None

        ep_idxs, t_idxs = self.get_sample_idxs(batch_size)
        s = [
            self.data['is_holding'][ep_idxs, t_idxs],
            self.data['in_hand'][ep_idxs, t_idxs],
            self.data['top_down'][ep_idxs, t_idxs],
            self.data['side_view'][ep_idxs, t_idxs],
            self.data['s_enc'][ep_idxs, t_idxs],
            self.data['sg_enc'][ep_idxs, t_idxs],
            self.data['g_enc'][ep_idxs, 0],
        ]
        a = self.data['actions'][ep_idxs, t_idxs]
        sp = [
            self.data['is_holding'][ep_idxs, t_idxs+1],
            self.data['in_hand'][ep_idxs, t_idxs+1],
            self.data['top_down'][ep_idxs, t_idxs+1],
            self.data['side_view'][ep_idxs, t_idxs+1],
            self.data['s_enc'][ep_idxs, t_idxs+1],
            self.data['sg_enc'][ep_idxs, t_idxs],
            self.data['g_enc'][ep_idxs, 0],
        ]
        rewards = self.data['sg_rewards'][ep_idxs, t_idxs]

        dones = self.data['dones'][ep_idxs, t_idxs]
        meta_rewards = self.reward_function(sp[4], sp[6], dones, sp[0])

        return s, a, sp, rewards, meta_rewards, dones

class GlobalReplayMemory(BaseMemory):
    def sample(self, batch_size):
        if len(self) < 10*batch_size:
            return None

        ep_idxs, t_idxs = self.get_sample_idxs(batch_size)
        s = [
            self.data['is_holding'][ep_idxs, t_idxs],
            self.data['in_hand'][ep_idxs, t_idxs],
            self.data['top_down'][ep_idxs, t_idxs],
            self.data['side_view'][ep_idxs, t_idxs],
            self.data['s_enc'][ep_idxs, t_idxs],
            self.data['g_enc'][ep_idxs, 0]
        ]
        a = self.data['actions'][ep_idxs, t_idxs]
        sp = [
            self.data['is_holding'][ep_idxs, t_idxs+1],
            self.data['in_hand'][ep_idxs, t_idxs+1],
            self.data['top_down'][ep_idxs, t_idxs+1],
            self.data['side_view'][ep_idxs, t_idxs+1],
            self.data['s_enc'][ep_idxs, t_idxs+1],
            self.data['g_enc'][ep_idxs, 0]
        ]
        dones = self.data['dones'][ep_idxs, t_idxs]
        rewards = self.reward_function(sp[-2], sp[-1],
                                       dones,
                                       sp[0])

        return s, a, sp, rewards, dones

class StepwiseReplayMemory(BaseMemory):
    def sample(self, batch_size):
        if len(self) < 10*batch_size:
            return None

        ep_idxs, t_idxs = self.get_sample_idxs(batch_size)
        s = [
            self.data['is_holding'][ep_idxs, t_idxs],
            self.data['in_hand'][ep_idxs, t_idxs],
            self.data['top_down'][ep_idxs, t_idxs],
            self.data['side_view'][ep_idxs, t_idxs],
            self.data['s_enc'][ep_idxs, t_idxs],
            self.data['g_enc'][ep_idxs, 0]
        ]
        a = self.data['actions'][ep_idxs, t_idxs]
        sp = [
            self.data['is_holding'][ep_idxs, t_idxs+1],
            self.data['in_hand'][ep_idxs, t_idxs+1],
            self.data['top_down'][ep_idxs, t_idxs+1],
            self.data['side_view'][ep_idxs, t_idxs+1],
            self.data['s_enc'][ep_idxs, t_idxs+1],
            self.data['g_enc'][ep_idxs, 0]
        ]
        dones = self.data['dones'][ep_idxs, t_idxs]
        rewards = self.data['sg_reached'][ep_idxs, t_idxs]

        return s, a, sp, rewards, dones

class HERMemory(BaseMemory):
    p_replay = 0.4
    def sample(self, batch_size):
        if len(self) < 10*batch_size:
            return None

        ep_idxs, t_idxs = self.get_sample_idxs(batch_size)
        ep_lengths = self.data['n_steps'][ep_idxs]
        future_offset = np.random.uniform(size=batch_size) * (ep_lengths - t_idxs)
        future_offset = future_offset.astype(int)
        goal_t_idxs = t_idxs + 1 + future_offset

        replay_mask = np.random.uniform(size=batch_size) < self.p_replay
        achieved_s_enc = self.data['s_enc'][ep_idxs, goal_t_idxs][replay_mask].copy()
        # goals should have distractor_flag=False
        achieved_s_enc[:,-1] = 0

        s = [
            self.data['is_holding'][ep_idxs, t_idxs],
            self.data['in_hand'][ep_idxs, t_idxs],
            self.data['top_down'][ep_idxs, t_idxs],
            self.data['side_view'][ep_idxs, t_idxs],
            self.data['s_enc'][ep_idxs, t_idxs],
            self.data['g_enc'][ep_idxs, 0]
        ]
        a = self.data['actions'][ep_idxs, t_idxs]
        sp = [
            self.data['is_holding'][ep_idxs, t_idxs+1],
            self.data['in_hand'][ep_idxs, t_idxs+1],
            self.data['top_down'][ep_idxs, t_idxs+1],
            self.data['side_view'][ep_idxs, t_idxs+1],
            self.data['s_enc'][ep_idxs, t_idxs+1],
            self.data['g_enc'][ep_idxs, 0]
        ]

        s[-1][replay_mask] = achieved_s_enc.copy()
        sp[-1][replay_mask] = achieved_s_enc.copy()

        dones = self.data['dones'][ep_idxs, t_idxs]

        rewards = self.reward_function(sp[-2], sp[-1],
                                       dones,
                                       sp[0])

        return s, a, sp, rewards, dones

class NeighborMemory(BaseMemory):
    p_replay = 0.8
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.g_space = kwargs['goal_space']

    def sample(self, batch_size):
        if len(self) < 10*batch_size:
            return None

        ep_idxs, t_idxs = self.get_sample_idxs(batch_size)

        relabel_mask = np.random.uniform(size=batch_size) < self.p_replay
        relabel_s_enc = self.data['s_enc'][ep_idxs, t_idxs][relabel_mask]
        relabel_g_enc = self.data['g_enc'][ep_idxs, 0][relabel_mask]

        # goals should have distractor_flag=False
        states = [self.g_space.all_structures[i] for i in relabel_s_enc[:,0]]
        goals = [self.g_space.all_structures[i] for i in relabel_g_enc[:,0]]

        relabeled_goals = [self.g_space.get_extended_subgoal(s, g)
                           for s,g in zip(states, goals)]
        # relabeled_goals = [self.g_space.get_subgoal(s, g)
                           # for s,g in zip(states, goals)]

        relabeled_g_enc = np.zeros_like(relabel_s_enc)
        relabeled_g_enc[:,0] = [self.g_space.structure_dict[g] for g in relabeled_goals]

        s = [
            self.data['is_holding'][ep_idxs, t_idxs],
            self.data['in_hand'][ep_idxs, t_idxs],
            self.data['top_down'][ep_idxs, t_idxs],
            self.data['side_view'][ep_idxs, t_idxs],
            self.data['s_enc'][ep_idxs, t_idxs],
            self.data['g_enc'][ep_idxs, 0]
        ]
        a = self.data['actions'][ep_idxs, t_idxs]
        sp = [
            self.data['is_holding'][ep_idxs, t_idxs+1],
            self.data['in_hand'][ep_idxs, t_idxs+1],
            self.data['top_down'][ep_idxs, t_idxs+1],
            self.data['side_view'][ep_idxs, t_idxs+1],
            self.data['s_enc'][ep_idxs, t_idxs+1],
            self.data['g_enc'][ep_idxs, 0]
        ]

        s[-1][relabel_mask] = relabeled_g_enc.copy()
        sp[-1][relabel_mask] = relabeled_g_enc.copy()

        dones = self.data['dones'][ep_idxs, t_idxs]

        rewards = self.reward_function(sp[-2], sp[-1],
                                       dones,
                                       sp[0])

        return s, a, sp, rewards, dones

def create_memory(*args, **kwargs):
    label_method = kwargs.pop('label_method')
    memory_class = {'subgoal' : LocalReplayMemory,
                    'finalgoal' : GlobalReplayMemory,
                    'stepwise' : StepwiseReplayMemory,
                    'her' : HERMemory,
                    'neighbor' : NeighborMemory,
                    'meta' : MetaMemory,
                    }[label_method]
    return memory_class(*args, **kwargs)
