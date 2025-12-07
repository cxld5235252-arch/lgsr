import numpy as np
import torch as th
from loguru import logger
from .gaussian_diffusion import GaussianDiffusion
from torch.cuda.amp import autocast
import torch
import torch.nn as nn
import torch.nn.functional as F


# class SelfAttention(nn.Module):
#     def __init__(self, k, heads=8):
#         super().__init__()
#         self.k, self.heads = k, heads

#         self.tokeys = nn.Linear(k, k * heads, bias=False)
#         self.toqueries = nn.Linear(k, k * heads, bias=False)
#         self.tovalues = nn.Linear(k, k * heads, bias=False)

#         self.unifyheads = nn.Linear(heads * k, k)

#     def forward(self, x):
#         b, k, H, w = x.size()
#         x = x.permute(0,2,3,1).view(b,-1,k)
#         b, t, k = x.size()
#         h = self.heads

#         queries = self.toqueries(x).view(b, t, h, k)
#         keys = self.tokeys(x).view(b, t, h, k)
#         values = self.tovalues(x).view(b, t, h, k)

#         keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
#         queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
#         values = values.transpose(1, 2).contiguous().view(b * h, t, k)

#         dot = torch.bmm(queries, keys.transpose(1, 2))
#         dot = F.softmax(dot, dim=2)

#         out = torch.bmm(dot, values).view(b, h, t, k)
#         out = out.transpose(1, 2).contiguous().view(b, t, h * k)
#         out = self.unifyheads(out)
#         out = out.permute(0,2,1).view(b,k,H,w)
#         return out

# class TeleAndXtRelation(nn.Module):
#     def __init__(self, img_size):
#         super(TeleAndXtRelation, self).__init__()
#         self.img_size = img_size
        
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.silu = nn.SiLU()
#         self.att1 = SelfAttention(64)
        
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.att2 = SelfAttention(128)

#         self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.att3 = SelfAttention(256)

#         # Decoder part (Symmetric to Encoder)
#         self.convtrans1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.bn4 = nn.BatchNorm2d(128)
#         self.att4 = SelfAttention(128)

#         self.convtrans2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.bn5 = nn.BatchNorm2d(64)
#         self.att5 = SelfAttention(64)

#         self.convtrans3 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.bn6 = nn.BatchNorm2d(6)

#     def forward(self, tele, xt):
#         # x = torch.cat((tele, xt), dim=1)
        
#         # Encoding path
#         x = self.silu(self.bn1(self.conv1(tele)))
#         x = self.silu(self.bn2(self.conv2(x)))
#         x = self.silu(self.bn3(self.conv3(x)))
#         # Decoding path
#         x = self.silu(self.bn4(self.convtrans1(x)))
#         x = self.silu(self.bn5(self.convtrans2(x)))
#         x = self.convtrans3(x)  # No activation after the last layer

#         return x
import torch
import torch.nn.functional as F

class BicubicUNet(torch.nn.Module):
    def __init__(self):
        super(BicubicUNet, self).__init__()
    
    def forward(self, x, y):
        # 原始图像尺寸
        original_size = x.size()[2:]

        # 下采样路径
        x1 = F.interpolate(x, scale_factor=1/2, mode='bicubic', align_corners=False)
        x2 = F.interpolate(x1, scale_factor=1/2, mode='bicubic', align_corners=False)
        x3 = F.interpolate(x2, scale_factor=1/2, mode='bicubic', align_corners=False)
        # x4 = F.interpolate(x3, scale_factor=1/2, mode='bicubic', align_corners=False)  # 最终的下采样结果是原始尺寸的1/16
        
        # 上采样路径
        # x = F.interpolate(x4, scale_factor=2, mode='bicubic', align_corners=False)
        x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)  # 恢复到原始尺寸
        
        return x

class TeleAndXtRelation(nn.Module):
    def __init__(self, img_size):
        super(TeleAndXtRelation, self).__init__()
        # 下采样模块
        self.downsample1 = nn.Sequential(
            nn.Conv2d(6, 12, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(12),
            nn.SiLU()
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(12, 48, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(48),
            nn.SiLU()
        )
        
        # 多头注意力机制
        self.multihead_attn = nn.MultiheadAttention(embed_dim=48, num_heads=4)
        
        # 上采样模块
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(48, 12, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(12),
            nn.SiLU()
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(12, 3, kernel_size=4, stride=4, padding=0),
            nn.BatchNorm2d(3),
            # nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, tele, xt):
        # 下采样
        x = torch.cat((tele, xt), dim=1)
        x = self.downsample1(x)
        x = self.downsample2(x)

        # 调整形状以适应多头注意力输入
        batch_size, channels, height, width = x.size()
        # x = x.permute(2, 3, 0, 1).reshape(height * width, batch_size, channels)

        # 多头注意力
        # attn_output, _ = self.multihead_attn(x, x, x)

        # 还原形状
        # attn_output = attn_output.reshape(height, width, batch_size, channels).permute(2, 3, 0, 1)

        # 上采样
        x = self.upsample1(x)
        x = self.upsample2(x)

        return x

# class TeleAndXtRelation(nn.Module):
#     def __init__(self, img_size):
#         super(TeleAndXtRelation, self).__init__()
#         self.img_size = img_size
        
#         self.conv1 = nn.Conv2d(6, 32, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.relu = nn.ReLU()

#         self.conv2 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(3)

#         self.res_conv = nn.Conv2d(6, 3, kernel_size=1, stride=1) 

#         # self.attention = nn.MultiheadAttention(embed_dim=3, num_heads=3, batch_first=True)
        
#         # self.conv3 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

#         # self.gate_generator = GateGenerator(256, 3, img_size)

#     def forward(self, tele, xt):
#         # perm_h, perm_w = torch.randperm(self.img_size), torch.randperm(self.img_size)
#         # tele = tele[:,:,perm_h,:][:,:,:,perm_w]
#         # perm_h, perm_w = torch.randperm(self.img_size), torch.randperm(self.img_size)
#         # xt = xt[:,:,perm_h,:][:,:,:,perm_w]
#         combined_feat = torch.cat((tele, xt), dim=1)
        
#         x = self.relu(self.bn1(self.conv1(combined_feat)))
#         x = self.relu(self.bn2(self.conv2(x)))

#         residual = self.res_conv(combined_feat)
#         x = x + residual

        
#         # b, c, h, w = x.size()
#         # x = x.view(b, c, h * w).permute(0, 2, 1)  # 调整为(batch, seq_len, embed_dim)的形状
#         # attn_output, _ = self.attention(x, x, x)
#         # attn_output = attn_output.permute(0, 2, 1).view(b, c, h, w)  

#         # x = self.conv3(attn_output)
#         return x


def space_timesteps(num_timesteps, sample_timesteps):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: timesteps for sampling
    :return: a set of diffusion steps from the original process to use.
    """
    all_steps = [int((num_timesteps/sample_timesteps) * x) for x in range(sample_timesteps)]
    
    return set(all_steps)

class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["sqrt_etas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        new_sqrt_etas = []
        for ii, etas_current in enumerate(base_diffusion.sqrt_etas):
            if ii in self.use_timesteps:
                new_sqrt_etas.append(etas_current)
                self.timestep_map.append(ii)
        kwargs["sqrt_etas"] = np.array(new_sqrt_etas)
    #     {'sqrt_etas': array([0.02      , 0.11716508, 0.17630456, 0.23362892, 0.29157413,
    #    0.35100928, 0.41235614, 0.47585773, 0.54167168, 0.60990993,
    #    0.68065802, 0.75398526, 0.82995066, 0.90860639, 0.99      ]), 'kappa': 2.0, 
    #     'model_mean_type': <ModelMeanType.START_X: 1>, 
    #     'loss_type': <LossType.MSE: 1>, 'scale_factor': 1.0, 
    #     'normalize_input': True, 'sf': 4, 'latent_flag': True}
        super().__init__(**kwargs)
        # self.weight_mask = nn.Parameter(torch.ones(2,3,256,256)).to('cuda:0')
        self.inner_size = 192
        self.start_idx = (256 - self.inner_size) // 2
        self.end_idx = self.start_idx + self.inner_size
        # self.inner_area = self.inner_size * self.inner_size
        # self.outer_area = 256*256 - self.inner_area
        
        # self.inner_mask = torch.zeros((4,3,256,256)).to('cuda:0')
        # self.inner_mask[:,:,self.start_idx:self.end_idx, self.start_idx:self.end_idx] = 1.
        # self.outer_mask = torch.ones((4,3,256,256)).to('cuda:0')
        # self.outer_mask[:,:,self.start_idx:self.end_idx, self.start_idx:self.end_idx] = 0.
        # self.setup_tele_prior(kwargs)    
        
        
    def setup_tele_prior(self,kwargs):
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0.0)  # 将权重初始化为0
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)  # 如果有偏置，则也初始化为0
        
        # self.tele_xt_score_map_net =  TeleAndXtRelation(kwargs['image_size']).cuda()
        # self.tele_xt_score_map_net = BicubicUNet().cuda()
        
        # self.tele_xt_score_map_net.apply(weights_init)
        self.tele = None

    def p_mean_variance(self, model, *args, **kwargs):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timestep_map, self.original_num_steps)

class _WrappedModel:
    def __init__(self, model, timestep_map, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        return self.model(x, new_ts, **kwargs)

# class SpacedDiffusionDDPM(GaussianDiffusionDDPM):
    # """
    # A diffusion process which can skip steps in a base diffusion process.

    # :param use_timesteps: a collection (sequence or set) of timesteps from the
    #                       original diffusion process to retain.
    # :param kwargs: the kwargs to create the base diffusion process.
    # """

    # def __init__(self, use_timesteps, **kwargs):
        
    #     self.use_timesteps = set(use_timesteps)
    #     self.timestep_map = []
    #     self.original_num_steps = len(kwargs["betas"])

    #     base_diffusion = GaussianDiffusionDDPM(**kwargs)  # pylint: disable=missing-kwoa
    #     last_alpha_cumprod = 1.0
    #     new_betas = []
    #     for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
    #         if i in self.use_timesteps:
    #             new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
    #             last_alpha_cumprod = alpha_cumprod
    #             self.timestep_map.append(i)
    #     kwargs["betas"] = np.array(new_betas)
    #     logger.info(kwargs)
    #     super().__init__(**kwargs)

    # def p_mean_variance(self, model, *args, **kwargs):  # pylint: disable=signature-differs
    #     return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    # def training_losses(self, model, *args, **kwargs):  # pylint: disable=signature-differs
    #     return super().training_losses(self._wrap_model(model), *args, **kwargs)

    # def _wrap_model(self, model):
    #     if isinstance(model, _WrappedModel):
    #         return model
    #     return _WrappedModel(model, self.timestep_map, self.original_num_steps)
    pass

