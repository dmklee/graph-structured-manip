import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn
import time

from src.unet_model import UNetXY

class QFunction():
    def __init__(self,
                 in_hand_shape,
                 top_down_shape,
                 enc_shape,
                 lr=0.0001,
                 gamma=0.9,
                 reward_style='positive',
                 target_update_freq=1000,
                 qmap_noise_range=(0.1,0),
                 random_action_range=(0.05,0),
                 include_s_enc=True,
                 log_fn=lambda s: print(s),
                 ):
        #unpack arguments
        self.log_fn = log_fn
        self.qmap_size = top_down_shape[-1]
        self.lr = lr
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.qmap_noise_range = qmap_noise_range
        self.random_action_range = random_action_range
        self.include_s_enc = include_s_enc

        assert reward_style in ('positive', 'negative', 'stepwise')
        self.reward_style = reward_style

        #set up networks
        self.network = UNetXY(in_hand_shape, top_down_shape,
                              enc_shape, include_s_enc=include_s_enc)
        self.target = UNetXY(in_hand_shape, top_down_shape,
                             enc_shape, include_s_enc=include_s_enc)

        self._hard_target_update()
        self.target.eval()
        self.optim = torch.optim.Adam(self.network.parameters(),
                                      lr=self.lr)

        self.opt_steps = 0
        self.last_qmap = None
        self.loss_history = []
        self.reward_history = []

    def log_progress(self, fdir, addendum=''):
        # save avg rewards
        rewards_data = np.vstack((np.arange(len(self.reward_history)),
                                 self.reward_history))
        np.save(f"{fdir}/q_function_rewards.npy", rewards_data)

        loss_data = np.vstack((np.arange(len(self.loss_history)),
                                 self.loss_history))
        np.save(f"{fdir}/q_function_loss.npy", loss_data)

        torch.save(self.network.state_dict(), f"{fdir}/q_function.pt")
        if addendum != '':
            torch.save(self.network.state_dict(), f"{fdir}/q_function{addendum}.pt")

    def action_selection(self, state, epsilon=0.0):
        if isinstance(epsilon, float):
            epsilon = np.full(len(state[0]), epsilon)
        is_holding = state[0].clone().cpu().numpy()

        actions = np.zeros((len(is_holding),3),dtype=int)
        actions[:,0] = is_holding[:,0]
        # use qmap+noise to select action
        qmap = self.get_qmap(state)
        self.last_qmap = qmap.copy()
        stdev = epsilon * self.qmap_noise_range[0] \
                + (1 - epsilon) * self.qmap_noise_range[1]
        stdev = np.broadcast_to(stdev[:,None,None], qmap.shape)
        qmap += np.random.normal(scale=stdev)
        self.last_qmap_noise = qmap.copy()
        x,y = np.unravel_index(qmap.reshape(len(qmap),-1).argmax(axis=1),
                               (self.qmap_size,self.qmap_size))
        actions[:,1] = x
        actions[:,2] = y
        p_random_action = epsilon * self.random_action_range[0] \
                            + (1 - epsilon) * self.random_action_range[1]

        for i, p in enumerate(p_random_action):
            if npr.random() < p:
                # random action
                actions[i,1:] = npr.randint(self.qmap_size, size=2)
        return actions

    def get_qmap(self, state):
        with torch.no_grad():
            qmap = self.network(*state)
            return qmap.cpu().numpy()

    def optimize(self, batch=None):
        if batch is None:
            return

        self.opt_steps += 1

        S, A, R, Sp, Done = batch
        R = R.squeeze()
        Done = Done.squeeze()
        batch_size = len(A)

        Qmap_pred = self.network(*S)
        Q_pred = Qmap_pred[np.arange(batch_size),A[:,1],A[:,2]]
        with torch.no_grad():
            Qmap_next_img = self.target(*Sp)
            Qmap_next = Qmap_next_img.view(batch_size,-1)
            # double Q update
            A_next = self.network(*Sp).view(batch_size,-1).max(1)[1].unsqueeze(1)
            Q_next = Qmap_next.gather(1,A_next).squeeze()
            if self.reward_style == 'positive':
                # Q_target = R + self.gamma * Q_next*(1-Done)
                Q_target = R + self.gamma * Q_next * (1-R)
            elif self.reward_style == 'negative':
                Q_target = (R-1) + self.gamma * Q_next * (1-R)
            elif self.reward_style == 'stepwise':
                Q_target = R + self.gamma * Q_next

        self.optim.zero_grad()
        loss = self.network.loss(Q_pred, Q_target)
        loss.backward()
        for param in self.network.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1,1)
        self.optim.step()

        if self.opt_steps > self.target_update_freq:
            self._hard_target_update()
            self.opt_steps = 0

        self.loss_history.append(loss.mean().item())

    def load_network(self, fdir):
        self.network.load_state_dict(torch.load(fdir+"/q_function.pt"))
        self._hard_target_update()

    def transfer_load_network(self, fname):
        # first initialize weights better
        for m in self.network.named_modules():
            if isinstance(m[1], nn.Linear):
                nn.init.xavier_normal_(m[1].weight.data)

        loaded_state_dict = torch.load(fname)

        my_state_dict = self.network.state_dict()
        for name, param in loaded_state_dict.items():
            if name not in my_state_dict:
                print('error name not in loaded')
                continue
            if param.size() == my_state_dict[name].size():
                my_state_dict[name].copy_(param)
            else:
                old_enc_size = param.size(1)//2
                new_enc_size = my_state_dict[name].size(1)//2
                # we need to copy state and goal separately since they both are extended
                my_state_dict[name][:,:old_enc_size].copy_(param[:,:old_enc_size])
                my_state_dict[name][:,new_enc_size:new_enc_size+old_enc_size].copy_(param[:,old_enc_size:])

        self.network.load_state_dict(my_state_dict)

        self._hard_target_update()

    def _hard_target_update(self):
        self.target.load_state_dict(self.network.state_dict())

    def to(self, device):
        self.network.to(device)
        self.target.to(device)
        return self

    def eval(self):
        self.network.eval()
