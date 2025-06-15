from __future__ import annotations
from einops import rearrange, repeat

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.upsample import UpSample
from monai.utils import InterpolateMode, UpsampleMode


class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, input_x):
        if self.data_format == "channels_last":
            return F.layer_norm(input_x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = input_x.mean(1, keepdim=True)
            s = (input_x - u).pow(2).mean(1, keepdim=True)
            input_x = (input_x - u) / torch.sqrt(s + self.eps)
            input_x = self.weight[:, None, None] * input_x + self.bias[:, None, None]
            return input_x
        return None

class Grouped_multi_axis_Hadamard_Product_Attention(nn.Module):
    # by-EGE-UNET
    def __init__(self, dim_in, x=8, y=8):
        super().__init__()

        c_dim_in = dim_in // 4
        k_size = 3
        pad = (k_size - 1) // 2

        self.params_xy = nn.Parameter(torch.Tensor(1, c_dim_in, x, y), requires_grad=True)
        nn.init.ones_(self.params_xy)
        self.conv_xy = nn.Sequential(nn.Conv2d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv2d(c_dim_in, c_dim_in, 1))

        self.params_zx = nn.Parameter(torch.Tensor(1, 1, c_dim_in, x), requires_grad=True)
        nn.init.ones_(self.params_zx)
        self.conv_zx = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.params_zy = nn.Parameter(torch.Tensor(1, 1, c_dim_in, y), requires_grad=True)
        nn.init.ones_(self.params_zy)
        self.conv_zy = nn.Sequential(nn.Conv1d(c_dim_in, c_dim_in, kernel_size=k_size, padding=pad, groups=c_dim_in),
                                     nn.GELU(), nn.Conv1d(c_dim_in, c_dim_in, 1))

        self.dw = nn.Sequential(
            nn.Conv2d(c_dim_in, c_dim_in, 1),
            nn.GELU(),
            nn.Conv2d(c_dim_in, c_dim_in, kernel_size=3, padding=1, groups=c_dim_in)
        )

        self.norm1 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.norm2 = LayerNorm(dim_in, eps=1e-6, data_format='channels_first')
        self.ldw = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
            nn.GELU(),
            nn.Conv2d(dim_in, dim_in, 1),
        )

    def forward(self, x):
        x = self.norm1(x)
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        params_xy = self.params_xy
        x1 = x1 * self.conv_xy(F.interpolate(params_xy, size=x1.shape[2:4], mode='bilinear', align_corners=True))
        x2 = x2.permute(0, 3, 1, 2)
        params_zx = self.params_zx
        x2 = x2 * self.conv_zx(
            F.interpolate(params_zx, size=x2.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x2 = x2.permute(0, 2, 3, 1)
        x3 = x3.permute(0, 2, 1, 3)
        params_zy = self.params_zy
        x3 = x3 * self.conv_zy(
            F.interpolate(params_zy, size=x3.shape[2:4], mode='bilinear', align_corners=True).squeeze(0)).unsqueeze(0)
        x3 = x3.permute(0, 2, 1, 3)
        x4 = self.dw(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.norm2(x)
        x = self.ldw(x)
        return x

class THPAEncFR3(nn.Module):
    def __init__(self, in_channels, expr):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(in_channels // 2)
        self.GHPA_dim = Grouped_multi_axis_Hadamard_Product_Attention(in_channels // 2, in_channels // 2)
        self.norm2 = nn.InstanceNorm3d(in_channels)
        self.mlp = MlpChannel(in_channels, expr)

    def forward(self, input_x: Tensor, dummy_tensor=None):
        input_x, x_residual = torch.chunk(input_x, 2, dim=1)
        input_x = self.norm1(input_x)
        B, C, W, H, D = input_x.shape

        random_direction = torch.randint(0, 3, (1,)).item()
        if random_direction == 0:
            WHD_dim = rearrange(self.GHPA_dim(rearrange(input_x, "b c w h d -> (h b) c w d")),
                                "(h b) c w d -> b c w h d", b=B)
            x_re = rearrange(input_x, "b c w h d -> (h b) c w d").flip([0])
            rWHD_dim = rearrange(self.GHPA_dim(x_re), "(h b) c w d -> b c w h d", b=B).flip([0])
            WHD_dim = WHD_dim + rWHD_dim
        elif random_direction == 1:
            WHD_dim = rearrange(self.GHPA_dim(rearrange(input_x, "b c w h d -> (w b) c h d")),
                                "(w b) c h d -> b c w h d", b=B)
            x_re = rearrange(input_x, "b c w h d -> (w b) c h d").flip([0])
            rWHD_dim = rearrange(self.GHPA_dim(x_re), "(w b) c h d -> b c w h d", b=B).flip([0])
            WHD_dim = WHD_dim + rWHD_dim
        elif random_direction == 2:
            WHD_dim = rearrange(self.GHPA_dim(rearrange(input_x, "b c w h d -> (d b) c w h")),
                                "(d b) c w h -> b c w h d", b=B)
            x_re = rearrange(input_x, "b c w h d -> (d b) c w h").flip([0])
            rWHD_dim = rearrange(self.GHPA_dim(x_re), "(d b) c w h -> b c w h d", b=B).flip([0])
            WHD_dim = WHD_dim + rWHD_dim
        else:
            raise NotImplementedError
        input_x = torch.cat((WHD_dim, x_residual), dim=1)
        input_x = self.norm2(input_x)
        input_x = self.mlp(input_x)
        return input_x

class NormDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.InstanceNorm3d(in_channels)
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
    def forward(self,input, dummy_tensor=None):
        return self.proj(self.norm(input))

class Learnable_Res_Skip_UpRepr4(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_dims = 3):
        super().__init__()
        self.upc = Convolution(
            spatial_dims=spatial_dims, in_channels = in_channels, out_channels = out_channels, strides=1,
            kernel_size=1, bias=False, conv_only=True
        )
        self.ups = UpSample(
            spatial_dims=spatial_dims,
            in_channels=out_channels,
            out_channels=out_channels,
            scale_factor=2,
            mode=UpsampleMode.NONTRAINABLE,
            interp_mode=InterpolateMode.LINEAR,
            align_corners=False,
        )

        self.repr_mldw = nn.Sequential(Convolution(spatial_dims=spatial_dims, in_channels=out_channels,
                                                   out_channels=out_channels, strides=1,
                                                   kernel_size=3, bias=False, conv_only=True, groups=out_channels // 12),
                                       nn.GELU(),
                                       Convolution(spatial_dims=spatial_dims, in_channels=out_channels,
                                                   out_channels=out_channels,
                                                   strides=1, kernel_size=1, bias=False, conv_only=True, groups=1)
                                       )

        self.norm = nn.InstanceNorm3d(out_channels)
        self.group_skip_scale = nn.Parameter(torch.Tensor(1, out_channels, 1, 1, 1), requires_grad=True)
        nn.init.ones_(self.group_skip_scale)
        self.group_res_scale = nn.Parameter(torch.Tensor(1), requires_grad=True)
        nn.init.ones_(self.group_res_scale)

    def forward(self, inp_skip, dummy_tensor=None):
        input, skip = inp_skip
        input = self.ups(self.upc(input))
        input = input + skip * self.group_skip_scale
        res = input

        input = self.norm(input)
        out = self.repr_mldw(input)

        return out + res * self.group_res_scale

class MlpChannel(nn.Module):
    def __init__(self, in_channels, expr = 1, out_channels = None):
        if out_channels is None:
            out_channels = in_channels
        super().__init__()
        self.fc1 = nn.Conv3d(in_channels, in_channels * expr, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(in_channels * expr, out_channels, 1)

    def forward(self, input_x):
        input_x = self.fc1(input_x)
        input_x = self.act(input_x)
        input_x = self.fc2(input_x)
        return input_x

def block_creator(coder_str, depths_unidirectional, in_channels, out_channels=0):
    if out_channels == 0:
        out_channels = in_channels

    if coder_str == "NormDownsample":
        block = NormDownsample(in_channels, out_channels)
    elif coder_str == "THPAEncFR3":
        block = nn.Sequential(*[
            THPAEncFR3(in_channels,expr=2)
            for _ in range(depths_unidirectional)
        ])
    elif coder_str == "Learnable_Res_Skip_UpRepr4":
        block = Learnable_Res_Skip_UpRepr4(in_channels,out_channels)
    else:
        print("encoder error")
        raise NotImplementedError
    return block

class JCMNetv8Enc(nn.Module):
    def __init__(self,
                 init_channels=4,
                 n_channels=32,
                 class_nums=4,
                 checkpoint_style="",
                 expr=2,
                 depths_unidirectional=None,
                 ):
        super(JCMNetv8Enc, self).__init__()

        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        else:
            self.outside_block_checkpointing = False

        if depths_unidirectional is None:
            raise NotImplementedError
        elif depths_unidirectional == "small":
            depths_unidirectional = [1, 1, 2, 2, 2]
        elif depths_unidirectional == "medium":
            depths_unidirectional = [3, 4, 4, 4, 4]
        elif depths_unidirectional == "large":
            depths_unidirectional = [3, 4, 8, 8, 8]

        encoder = ["THPAEncFR3", "THPAEncFR3", "THPAEncFR3", "THPAEncFR3", "THPAEncFR3"]


        downcoder = "NormDownsample"

        self.stem = nn.Conv3d(init_channels, n_channels, kernel_size=1)

        self.repr_block_0 = block_creator(encoder[0], depths_unidirectional[0], n_channels)
        self.dwn_block_0 = block_creator(downcoder, 1, n_channels, n_channels * 2)

        self.repr_block_1 = block_creator(encoder[1], depths_unidirectional[1], n_channels * 2)
        self.dwn_block_1 = block_creator(downcoder, 1, n_channels * 2, n_channels * 4)

        self.repr_block_2 = block_creator(encoder[2], depths_unidirectional[2], n_channels * 4)
        self.dwn_block_2 = block_creator(downcoder, 1, n_channels * 4, n_channels * 8)

        self.repr_block_3 = block_creator(encoder[3], depths_unidirectional[3], n_channels * 8)
        self.dwn_block_3 = block_creator(downcoder, 1, n_channels * 8, n_channels * 16)

        self.emb_block = block_creator(encoder[4], depths_unidirectional[4], n_channels * 16)


        if self.outside_block_checkpointing:
        # Used to fix PyTorch checkpointing bug from MedNeXt
            self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

    # by-MedNeXt
    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor, use_reentrant=True)
        return x

    def forward(self, input: Tensor):
        if self.outside_block_checkpointing:

            pass
        else:
            input = self.stem(input)
            skips = []
            repr0 = self.repr_block_0(input)
            dwn0 = self.dwn_block_0(repr0)
            skips.append(repr0)
            del repr0

            repr1 = self.repr_block_1(dwn0)
            dwn1 = self.dwn_block_1(repr1)
            skips.append(repr1)
            del repr1

            repr2 = self.repr_block_2(dwn1)
            dwn2 = self.dwn_block_2(repr2)
            skips.append(repr2)
            del repr2

            repr3 = self.repr_block_3(dwn2)
            dwn3 = self.dwn_block_3(repr3)
            skips.append(repr3)
            del repr3

            hidden = self.emb_block(dwn3)

            return hidden, tuple(skips)

class JCMNetv8Dec(nn.Module):
    def __init__(self,
                 init_channels=4,
                 n_channels=32,
                 class_nums=4,
                 checkpoint_style="",
                 expr=2,
                 depths_unidirectional=None,
                 ):
        super(JCMNetv8Dec, self).__init__()

        if depths_unidirectional is None:
            raise NotImplementedError
        elif depths_unidirectional == "small":
            depths_unidirectional = [1, 1, 2, 2, 2]
        elif depths_unidirectional == "medium":
            depths_unidirectional = [3, 4, 4, 4, 4]
        elif depths_unidirectional == "large":
            depths_unidirectional = [3, 4, 8, 8, 8]

        if checkpoint_style == 'outside_block':
            self.outside_block_checkpointing = True
        else:
            self.outside_block_checkpointing = False

        decoder = ["Learnable_Res_Skip_UpRepr4","Learnable_Res_Skip_UpRepr4",
                   "Learnable_Res_Skip_UpRepr4","Learnable_Res_Skip_UpRepr4"]

        self.repr_block_up_3 = block_creator(decoder[3],depths_unidirectional[3],n_channels * 16,n_channels * 8)
        self.repr_block_up_2 = block_creator(decoder[2],depths_unidirectional[2],n_channels * 8,n_channels * 4)
        self.repr_block_up_1 = block_creator(decoder[1],depths_unidirectional[1],n_channels * 4,n_channels * 2)
        self.repr_block_up_0 = block_creator(decoder[0],depths_unidirectional[0],n_channels * 2,n_channels)

        if self.outside_block_checkpointing:
        # Used to fix PyTorch checkpointing bug from MedNeXt
            self.dummy_tensor = nn.Parameter(torch.tensor([1.]), requires_grad=True)

    # by-MedNeXt
    def iterative_checkpoint(self, sequential_block, x):
        """
        This simply forwards x through each block of the sequential_block while
        using gradient_checkpointing. This implementation is designed to bypass
        the following issue in PyTorch's gradient checkpointing:
        https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
        """
        for l in sequential_block:
            x = checkpoint.checkpoint(l, x, self.dummy_tensor, use_reentrant=True)
        return x

    def forward(self, hidden, skips):
        if self.outside_block_checkpointing:
            pass

        else:
            dec = self.repr_block_up_3((hidden,skips[3]))
            dec = self.repr_block_up_2((dec,skips[2]))
            dec = self.repr_block_up_1((dec,skips[1]))
            dec = self.repr_block_up_0((dec,skips[0]))

            return dec

class NormalU_Net(nn.Module):
    def __init__(self,
                 init_channels = 4,
                 n_channels = 24,
                 class_nums = 4,
                 checkpoint_style = "",
                 expr = 2,
                 depths_unidirectional=None,
                 ):
        super().__init__()
        args_list = [init_channels,
                     n_channels,
                     class_nums,
                     checkpoint_style,
                     expr,
                     depths_unidirectional]
        self.ParallelU_Net_enc_m = JCMNetv8Enc(*args_list)
        self.ParallelU_Net_dec_m = JCMNetv8Dec(*args_list)

        self.norm = nn.GroupNorm(n_channels, n_channels)
        self.proj = MlpChannel(n_channels, expr, class_nums)

    def forward(self, input, dummy_tensor=None):
        hidden_m, skips_m = self.ParallelU_Net_enc_m(input)
        out = self.ParallelU_Net_dec_m(hidden_m, skips_m)
        out = self.proj(self.norm(out))
        return out


if __name__ == '__main__':
    cuda0 = torch.device('cuda:0')
    x = torch.rand((2, 4, 128, 128, 128), device=cuda0)
    model = NormalU_Net(depths_unidirectional='small')
    model.cuda()
    print(model)
    print(str(sum([param.nelement() for param in model.parameters()]) / 1e6) + 'M')
