import numpy as np
import torch
from multiprocessing import Process, Pipe
import os

def worker(remote, parent_remote, env_fn):
    '''
    Worker function which interacts with the environment over remote

    Args:
    - remote: Worker remote connection
    - parent_remote: RL EnvRunner remote connection
    - env_fn: Function which creates a deictic environment
    '''
    parent_remote.close()
    env = env_fn()

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                obs, s_enc, done = env.step(data)
                remote.send((obs, s_enc, done))
            elif cmd == 'reset':
                obs, s_enc, g_enc = env.reset(*data)
                remote.send((obs, s_enc, g_enc))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_subgoal':
                s = env.get_subgoal(data)
                remote.send(s)
            elif cmd == 'is_terminal_goal':
                s = env.goal_space.is_terminal(env.goal)
                remote.send(s)
            elif cmd == 'opt_path_length':
                s = env.opt_path_length
                remote.send(s)
            elif cmd == 'enc_to_desc':
                s = env.enc_to_desc(data)
                remote.send(s)
            elif cmd == 'random_action':
                s = env.random_action()
                remote.send(s)
            elif cmd == 'get_desc':
                s = tuple(env.structure_desc)
                remote.send(s)
            elif cmd == 'get_path':
                s = env.planner.get_path(*data)
                remote.send(s)
    except KeyboardInterrupt:
        print('EnvRunner worker: caught keyboard interrupt')

class RLRunner(object):
    '''
    RL environment runner which runs mulitpl environemnts in parallel in subprocesses
    and communicates with them via pipe

    Args:
    - envs: List of DeiciticEnvs
    '''
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False

        self.num_envs = len(env_fns)
        self.remotes, self.worker_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
        self.processes = [Process(target=worker, args=(worker_remote, remote, env_fn))
                          for (worker_remote, remote, env_fn) in zip(self.worker_remotes, self.remotes, env_fns)]
        self.num_processes = len(self.processes)

        for process in self.processes:
            process.daemon = True
            process.start()
        for remote in self.worker_remotes:
            remote.close()

    def get_subgoals(self, s_enc):
        [r.send(('get_subgoal', s_enc[i])) for (i,r) in enumerate(self.remotes)]
        return np.stack([r.recv() for r in self.remotes],axis=0)

    def get_some_subgoals(self, env_nums, s_enc):
        [self.remotes[n].send(('get_subgoal', s_enc[i])) for (i,n) in enumerate(env_nums)]
        return np.stack([self.remotes[n].recv() for n in env_nums],axis=0)

    def random_action(self):
        [r.send(('random_action', None)) for (i,r) in enumerate(self.remotes)]
        return np.stack([r.recv() for r in self.remotes])

    def is_terminal_goal(self, env_num):
        self.remotes[env_num].send(('is_terminal_goal', None))
        return self.remotes[env_num].recv()

    def opt_path_length(self, env_nums):
        [self.remotes[i].send(('opt_path_length', None)) for i in env_nums]
        return np.array([self.remotes[i].recv() for i in env_nums])

    def get_path(self, start, goal):
        if isinstance(start, str):
            start = [start]
        if isinstance(goal, str):
            goal = [goal]
        for i in range(len(start)):
            self.remotes[i].send(('get_path', (start[i], goal[i])))
        return [self.remotes[i].recv() for i in range(len(start))]

    def step(self, actions, env_nums=None):
        '''
        Step the environments synchronously.

        Args:
          - actions: PyTorch variable of environment actions
          - env_nums: list of env indices to step. Default is None, meaning all
          envs are stepped
        '''
        self.stepAsync(actions, env_nums)
        return self.stepWait(env_nums)

    def stepAsync(self, actions, env_nums=None):
        '''
        Step each environment in a async fashion

        Args:
          - actions: numpy array (n,3) dtype=int
          - env_nums: list of env indices to step. Default is None, meaning all
          envs are stepped
        '''
        if env_nums is None:
            env_nums = np.arange(self.num_envs)
        for i, env_num in enumerate(env_nums):
            self.remotes[env_num].send(('step', actions[i]))
        self.waiting = True

    def stepWait(self, env_nums=None):
        '''
        Wait until each environment has completed its next step

        Args:
          - env_nums: list of env indices to wait for. Default is None,
          meaning all envs are waited for

        Returns: (obs, dones)
          - obs: Torch vector of observations
          - s_encs: Torch vector of state encodings
          - dones: Numpy vector of 0/1 flags indicating if episode is done
        '''
        if env_nums is None:
            env_nums = np.arange(self.num_envs)

        results = [self.remotes[env_num].recv() for env_num in env_nums]
        self.waiting = False

        obs, s_encs, dones = zip(*results)
        obs = [np.stack(a,axis=0) for a in zip(*obs)]
        obs[0] = np.expand_dims(obs[0], axis=1)
        s_encs = np.stack(s_encs)
        return obs, s_encs, np.stack(dones)[...,None]

    def reset(self, goal_distribution='all', goal_subset=None):
        '''
        Reset each environment

        Returns:
            obs: tuple of np arrays for is_holding, in_hand, top_down, side_view
            s_encs: np array of state encodings
            g_encs: np array of goal encodings
        '''
        for remote in self.remotes:
            remote.send(('reset', (goal_distribution, goal_subset)))

        obs, s_encs, g_encs = zip(*[remote.recv() for remote in self.remotes])

        obs = [np.stack(a, axis=0) for a in zip(*obs)]
        obs[0] = np.expand_dims(obs[0], axis=1)
        return obs, np.stack(s_encs), np.stack(g_encs)

    def reset_envs(self, env_nums, goal_distribution='all', goal_subset=None):
        '''
        Use this to reset specific environments, in case only a subset are done
        '''
        for env_num in env_nums:
            self.remotes[env_num].send(('reset', (goal_distribution, goal_subset)))

        obs, s_encs, g_encs = zip(*[self.remotes[n].recv() for n in env_nums])

        obs = [np.stack(a) for a in zip(*obs)]
        obs[0] = np.expand_dims(obs[0], axis=1)
        return obs, np.stack(s_encs), np.stack(g_encs)

    def enc_to_desc(self, enc):
        [self.remotes[i].send(('enc_to_desc', enc[i])) for i in range(len(enc))]
        return [self.remotes[i].recv() for i in range(len(enc))]

    def close(self):
        '''
        Close all worker processes
        '''
        self.closed = True
        if self.waiting:
            [remote.recv() for remote in self.remotes]
        [remote.send(('close', None)) for remote in self.remotes]
        [process.join() for process in self.processes]

