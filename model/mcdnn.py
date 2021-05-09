# MCDNN from https://github.com/LqNoob/MelodyExtraction-MCDNN/blob/master/MelodyExtraction_SCDNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
class MCDNN(nn.Module):
    def __init__(self):
        super(MCDNN, self).__init__()

        self.mcdnn = nn.Sequential(
            nn.Linear(360 * 3, 2048),
            nn.Dropout(0.2),
            nn.SELU(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.2),
            nn.SELU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.SELU(),
            nn.Linear(512, 360)
        )
        self.bm_layer = nn.Sequential(
            nn.Linear(360 * 3, 512),
            nn.Dropout(0.2),
            nn.SELU(),
            nn.Linear(512, 128),
            nn.Dropout(0.2),
            nn.SELU(),
            nn.Linear(128, 1),
            nn.SELU()
        )

    def forward(self, x):
        # [bs, 3, f, t]
        x = x.view(x.shape[0], -1, x.shape[-1])
        x = x.permute(0,2,1) # [bs, t, f * 3]
        output_pre = self.mcdnn(x)
        bm = self.bm_layer(x)
        output_pre = output_pre.permute(0,2,1)
        output_pre = output_pre.unsqueeze(dim=1)
        bm = bm.permute(0,2,1)
        bm = bm.unsqueeze(dim=1)
        output_pre = torch.cat((bm, output_pre), dim=2)
        output = nn.Softmax(dim=2)(output_pre)

        return output, output_pre