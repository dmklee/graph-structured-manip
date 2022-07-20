import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tfm
import numpy as np

from src.model_utils import View, Flatten, Norm_dict

class SymbolicEncoder(nn.Module):
    def __init__(self,
                 img_size,
                 n_classes,
                 n_channels=16,
                 n_fc_units=128,
                 norm_type='instance',
                 lr= 0.001,
                 ):
        super().__init__()
        self.img_size = img_size
        self.n_classes = n_classes

        norm_fn = Norm_dict[norm_type]
        bias = norm_type == 'identity'
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, n_channels, 3, 2, bias=bias),
            norm_fn(n_channels),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_channels, n_channels*2, 3, 2, bias=bias),
            norm_fn(n_channels*2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_channels*2, n_channels * 4, 3, 2, bias=bias),
            norm_fn(n_channels*4),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_channels*4, n_channels*8, 3, 2, bias=bias),
            norm_fn(n_channels*8),
            nn.LeakyReLU(0.1, True),
            # nn.Conv2d(n_channels*8, n_channels*16, 3, 1, bias=bias),
            # norm_fn(n_channels*16),
            # nn.LeakyReLU(0.1, True),
            Flatten()
        )
        self.conv_output_size = np.product(self._get_conv_output_shape())
        print(self.conv_output_size)

        self.linear_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, n_fc_units),
            nn.ReLU(True),
            nn.Linear(n_fc_units, n_fc_units),
            nn.ReLU(True),
        )
        self.structure_output = nn.Linear(n_fc_units, self.n_classes)
        self.flag_output = nn.Linear(n_fc_units, 2)

        self.lr = lr
        params = list(self.conv_layers.parameters()) \
                    + list(self.linear_layers.parameters())
        self.optim = torch.optim.Adam(params, lr =self.lr)
        self.opt_count = 0
        self.loss_history = []
        self.accuracy_history = []

        self.random_crop = tfm.RandomCrop(self.img_size)
        self.random_resize = tfm.RandomResizedCrop(self.img_size,
                                                   scale=(0.8,1.0),
                                                   ratio=(0.85,1.17))
        self.blur = tfm.GaussianBlur(7, sigma=(0.1,1))

    def _get_conv_output_shape(self):
        t = torch.zeros((1, 1, self.img_size, self.img_size))
        return self.conv_layers(t).shape

    def forward(self, x):
        y = self.conv_layers(x)
        y = self.linear_layers(y)
        structure_pred = self.structure_output(y)
        flag_pred = self.flag_output(y)
        return structure_pred, flag_pred

    def calc_accuracy(self, obs, enc):
        pred_structure, pred_flag = self.forward(obs)

        loss, accuracy = self.loss_function(pred_structure, pred_flag, enc)
        return accuracy

    def loss_function(self, pred_structure, pred_flag, target):
        """Compute loss

        :returns:
            :loss: loss variable for backpropagation
            :accuracy: tensor of size L with the average number of correctly
            predicted layer labels (e.g. the argmax in pred equals the class
            in target)
        """
        criterion = nn.CrossEntropyLoss()

        # perform loss on layers
        target_structure = target[:,:-1]
        target_structure_idx = torch.max(target_structure, 1)[1]
        structure_loss = criterion(pred_structure, target_structure_idx)

        # perform loss on distractor flag
        target_flag_idx = target[:,-1].long()
        flag_loss = criterion(pred_flag, target_flag_idx)

        total_loss = structure_loss# + flag_loss

        # compute accuracy
        with torch.no_grad():
            pred_structure_idx = torch.max(pred_structure, 1)[1]
            pred_flag_idx = torch.max(pred_flag, 1)[1]
            accuracy = torch.cat([(pred_structure_idx== target_structure_idx).float().mean(0,keepdim=True),
                                  (pred_flag_idx == target_flag_idx).float().mean(0,keepdim=True)])

        return total_loss, accuracy

    def domain_randomize(self, obs):
        tfm.functional.pad(obs, padding=4, padding_mode='reflect')
        if np.random.random() < 0.5:
            obs = self.random_crop(obs)
        else:
            obs = self.random_resize(obs)
        obs = self.blur(obs)
        obs += 0.01*torch.randn(*obs.size(), dtype=torch.float32, device=obs.device)
        obs += 0.1*np.random.randn()

        return obs

    def optimize(self, obs, enc):
        #domain randomization
        with torch.no_grad():
            noised_obs = self.domain_randomize(obs)

        # import matplotlib.pyplot as plt
        # for i in range(20):
            # f, ax = plt.subplots(ncols=2)
            # ax[0].imshow(obs[i,0].cpu())
            # ax[1].imshow(noised_obs[i,0].cpu())
            # plt.show()
        # exit()


        self.optim.zero_grad()
        pred_structure, pred_flag = self.forward(noised_obs)

        loss, accuracy = self.loss_function(pred_structure, pred_flag, enc)
        self.loss_history.append(loss.item())
        self.accuracy_history.append(accuracy.cpu().numpy())

        loss.backward()
        self.optim.step()

    def load_network(self, fdir):
        self.load_state_dict(torch.load(fdir+"/sym_encoder.pt"))

    def log_progress(self, fdir, addendum=''):
        if len(self.loss_history) == 0:
            return

        loss_data = np.vstack((np.arange(len(self.loss_history)),
                                 self.loss_history)).T
        np.save(f"{fdir}/sym_encoder_loss.npy", loss_data)

        accuracy_data = np.concatenate((np.arange(len(self.accuracy_history))[:,None],
                                 self.accuracy_history),axis=1)
        np.save(f"{fdir}/sym_encoder_accuracy.npy", accuracy_data)

        torch.save(self.state_dict(), f"{fdir}/sym_encoder.pt")
        if addendum != '':
            torch.save(self.state_dict(), f"{fdir}/sym_encoder{addendum}.pt")
