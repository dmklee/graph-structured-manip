import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.model_utils import *

class UNetXY(nn.Module):
    def __init__(self,
                 in_hand_shape,
                 top_down_shape,
                 enc_shape,
                 task_sharing_mode='broadcast',
                 include_s_enc=True,
                ):
        super().__init__()
        self.top_down_shape = top_down_shape
        self.in_hand_shape = in_hand_shape
        self.enc_shape = enc_shape

        self.in_hand_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.ReLU(True),
            # nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3,padding=1),
            nn.ReLU(True),
        )

        # assert task_sharing_mode in ('broadcast', 'reshape')
        self.include_s_enc = include_s_enc
        inp_size = np.product(self.enc_shape)
        if self.include_s_enc:
            inp_size *= 2
        self.sym_enc_depth1 = 32
        self.sym_encoder1 = nn.Sequential(
            nn.Linear(inp_size, 256),
            nn.ReLU(True),
            nn.Linear(256, self.sym_enc_depth1),
            nn.ReLU(True),
            Broadcast2d(self.top_down_shape[2]//4),
        )
        self.sym_enc_depth2 = 64
        self.sym_encoder2 = nn.Sequential(
            nn.Linear(inp_size, 256),
            nn.ReLU(True),
            nn.Linear(256, self.sym_enc_depth2),
            nn.ReLU(True),
            Broadcast2d(self.top_down_shape[2]//16),
        )
        # elif task_sharing_mode == 'reshape':
            # self.sym_enc_depth = 16
            # self.sym_encoder = nn.Sequential(
                # nn.Linear(inp_size, 256),
                # nn.ReLU(True),
                # nn.Linear(256, self.sym_enc_depth*(self.top_down_shape[2]//16)**2),
                # nn.ReLU(True),
                # View((self.sym_enc_depth, self.top_down_shape[2]//16,
                      # self.top_down_shape[2]//16)),
            # )
        BIAS = False
        self.down1 = nn.Sequential(
            nn.Conv2d(self.top_down_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            BasicBlock(32, 32, dilation=1, bias=BIAS),
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            BasicBlock(32, 64,
                downsample=nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=1, bias=BIAS)
                ),dilation=1, bias=BIAS),
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            BasicBlock(64, 128,
                downsample=nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=1, bias=BIAS)
                ),dilation=1, bias=BIAS),
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            BasicBlock(128+self.sym_enc_depth1, 256,
                downsample=nn.Sequential(
                    nn.Conv2d(128+self.sym_enc_depth1, 256, kernel_size=1, bias=BIAS)
                ),dilation=1, bias=BIAS),
        )
        self.down5 = nn.Sequential(
            nn.MaxPool2d(2),
            BasicBlock(256,512,
                downsample=nn.Sequential(
                    nn.Conv2d(256,512, kernel_size=1, bias=BIAS)
                ),dilation=1, bias=BIAS),
            nn.Conv2d(512,256,kernel_size=1,bias=BIAS),
        )
        self.down6 = nn.Sequential(
                        BasicBlock(512+self.sym_enc_depth2,256,
                                   downsample=nn.Sequential(
                                       nn.Conv2d(512+self.sym_enc_depth2,256,
                                                 kernel_size=1,bias=BIAS),
                                   ),
                                   dilation=1, bias=BIAS
                                  )
        )

        self.up1 = nn.Sequential(
            BasicBlock(512,256,
                       downsample=nn.Sequential(
                           nn.Conv2d(512,256,kernel_size=1,bias=BIAS),
                       ),
                       dilation=1, bias=BIAS),
            nn.Conv2d(256,128,kernel_size=1,bias=BIAS),
        )
        self.up2 = nn.Sequential(
            BasicBlock(256,128,
                       downsample=nn.Sequential(
                           nn.Conv2d(256,128,kernel_size=1,bias=BIAS),
                       ),
                       dilation=1, bias=BIAS),
            nn.Conv2d(128,64,kernel_size=1,bias=BIAS),
        )
        self.up3 = nn.Sequential(
            BasicBlock(128,64,
                       downsample=nn.Sequential(
                           nn.Conv2d(128,64,kernel_size=1,bias=BIAS),
                       ),
                       dilation=1, bias=BIAS),
            nn.Conv2d(64,32,kernel_size=1,bias=BIAS),
        )
        self.up4 = nn.Sequential(
            BasicBlock(64,32,
                       downsample=nn.Sequential(
                           nn.Conv2d(64,32,kernel_size=1,bias=BIAS),
                       ),
                       dilation=1, bias=BIAS),
        )
        self.pick_q_values = nn.Conv2d(32,1,kernel_size=1,stride=1)
        self.place_q_values = nn.Conv2d(32,1,kernel_size=1,stride=1)

        self.criterion = nn.MSELoss()
        custom_initialize(self)

    def forward(self, is_holding, in_hand, top_down, obs, s_enc, g_enc):
        if self.include_s_enc:
            sg_enc = torch.cat((s_enc.view(s_enc.size(0),-1),
                                g_enc.view(g_enc.size(0),-1)),dim=1)
        else:
            sg_enc = g_enc.view(g_enc.size(0), -1)
        task_enc1 = self.sym_encoder1(sg_enc)
        task_enc2 = self.sym_encoder2(sg_enc)
        # downsampling to produce encoding
        x1 = self.down1(top_down)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(torch.cat((x3,task_enc1),dim=1))
        x5 = self.down5(x4)
        in_hand_enc = self.in_hand_encoder(in_hand)

        x5 = self.down6(torch.cat((x5, in_hand_enc, task_enc2),dim=1))

        #upsampling to produce q-map
        x4_up = self.up1(torch.cat((x4, F.interpolate(x5,size=x4.shape[-1],
                               mode='bilinear',align_corners=False)),dim=1))
        x3_up = self.up2(torch.cat((x3, F.interpolate(x4_up,size=x3.shape[-1],
                               mode='bilinear',align_corners=False)),dim=1))
        x2_up = self.up3(torch.cat((x2, F.interpolate(x3_up,size=x2.shape[-1],
                               mode='bilinear',align_corners=False)),dim=1))
        x1_up = self.up4(torch.cat((x1, F.interpolate(x2_up,size=x1.shape[-1],
                               mode='bilinear',align_corners=False)),dim=1))

        q_values = torch.cat((self.pick_q_values(x1_up),
                              self.place_q_values(x1_up)),dim=1)

        is_holding_exp = is_holding.unsqueeze(2).unsqueeze(3)
        is_holding_exp = is_holding_exp.repeat(1, 1,
                                               q_values.size(2),
                                               q_values.size(3))

        ret = torch.gather(q_values, 1, is_holding_exp).squeeze(1)
        return ret

    def loss(self, pred, target):
        return self.criterion(pred, target)
