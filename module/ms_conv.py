import torch
import torch.nn as nn
from timm.models.layers import DropPath
from spikingjelly.clock_driven.neuron import (
    MultiStepLIFNode,
    MultiStepParametricLIFNode,
)

import torch.nn.functional as F
import math
from inspect import isfunction

# codes adopted from https://github.com/lucidrains/mixture-of-experts/blob/master/mixture_of_experts/mixture_of_experts.py

# constants

MIN_EXPERT_CAPACITY = 4

# helper functions

def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val

def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)

# tensor related helper functions

def top1(t):
    values, index = t.topk(k=1, dim=-1)
    values, index = map(lambda x: x.squeeze(dim=-1), (values, index))
    return values, index

def cumsum_exclusive(t, dim=-1):
    num_dims = len(t.shape)
    num_pad_dims = - dim - 1
    pre_padding = (0, 0) * num_pad_dims
    pre_slice   = (slice(None),) * num_pad_dims
    padded_t = F.pad(t, (*pre_padding, 1, 0)).cumsum(dim=dim)
    return padded_t[(..., slice(None, -1), *pre_slice)]

# pytorch one hot throws an error if there are out of bound indices.
# tensorflow, in contrast, does not throw an error
def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    return F.one_hot(indexes, max(max_index + 1, max_length))[..., :max_length]

def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)

# activations

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_



class Erode(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)
        )

    def forward(self, x):
        return self.pool(x)


class MS_MLP_Expert(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        drop=0.0,
        spike_mode="lif",
        layer=0,
        tau=2.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.res = in_features == hidden_features
        self.fc1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.fc1_bn = nn.BatchNorm2d(hidden_features)
        if spike_mode == "lif":
            self.fc1_lif = MultiStepLIFNode(tau=tau, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc1_lif = MultiStepParametricLIFNode(
                init_tau=tau, detach_reset=True, backend="cupy"
            )

        self.fc2_conv = nn.Conv2d(
            hidden_features, out_features, kernel_size=1, stride=1
        )
        self.fc2_bn = nn.BatchNorm2d(out_features)
        if spike_mode == "lif":
            self.fc2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.fc2_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.c_hidden = hidden_features
        self.c_output = out_features
        self.layer = layer
        
    def reset(self):
        for m in self.modules():
            if m is self:
                continue
            if hasattr(m, "reset"):
                m.reset()

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        identity = x

        x = self.fc1_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc1_lif"] = x.detach()
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, H, W).contiguous()
        if self.res:
            x = identity + x
            identity = x
        x = self.fc2_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_fc2_lif"] = x.detach()
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        # x = x + identity
        return x, hook


class MS_SSA_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=1,
        mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.dvs = dvs
        self.num_heads = num_heads
        if dvs:
            self.pool = Erode()
        self.scale = 0.125
        self.q_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.q_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.k_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.k_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.v_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm2d(dim)
        if spike_mode == "lif":
            self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        elif spike_mode == "plif":
            self.v_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        if spike_mode == "lif":
            self.attn_lif = MultiStepLIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.attn_lif = MultiStepParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )

        self.talking_heads = nn.Conv1d(
            num_heads, num_heads, kernel_size=1, stride=1, bias=False
        )
        if spike_mode == "lif":
            self.talking_heads_lif = MultiStepLIFNode(
                tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.talking_heads_lif = MultiStepParametricLIFNode(
                init_tau=2.0, v_threshold=0.5, detach_reset=True, backend="cupy"
            )

        self.proj_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm2d(dim)

        if spike_mode == "lif":
            self.shortcut_lif = MultiStepLIFNode(
                tau=2.0, detach_reset=True, backend="cupy"
            )
        elif spike_mode == "plif":
            self.shortcut_lif = MultiStepParametricLIFNode(
                init_tau=2.0, detach_reset=True, backend="cupy"
            )

        self.mode = mode
        self.layer = layer

    def forward(self, x, hook=None):
        T, B, C, H, W = x.shape
        identity = x
        N = H * W
        x = self.shortcut_lif(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_first_lif"] = x.detach()

        x_for_qkv = x.flatten(0, 1)
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, H, W).contiguous()
        q_conv_out = self.q_lif(q_conv_out)

        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_q_lif"] = q_conv_out.detach()
        q = (
            q_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, H, W).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        if self.dvs:
            k_conv_out = self.pool(k_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_k_lif"] = k_conv_out.detach()
        k = (
            k_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, H, W).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        if self.dvs:
            v_conv_out = self.pool(v_conv_out)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_v_lif"] = v_conv_out.detach()
        v = (
            v_conv_out.flatten(3)
            .transpose(-1, -2)
            .reshape(T, B, N, self.num_heads, C // self.num_heads)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )  # T B head N C//h

        kv = k.mul(v)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv_before"] = kv
        if self.dvs:
            kv = self.pool(kv)
        kv = kv.sum(dim=-2, keepdim=True)
        kv = self.talking_heads_lif(kv)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_kv"] = kv.detach()
        x = q.mul(kv)
        if self.dvs:
            x = self.pool(x)
        if hook is not None:
            hook[self._get_name() + str(self.layer) + "_x_after_qkv"] = x.detach()

        x = x.transpose(3, 4).reshape(T, B, C, H, W).contiguous()
        x = (
            self.proj_bn(self.proj_conv(x.flatten(0, 1)))
            .reshape(T, B, C, H, W)
            .contiguous()
        )

        x = x + identity
        return x, v, hook



## The gating should select expert per batch in order to run MS_MLP_Expert that receives a shape of T, B, C, H, W
class Top2Gating(nn.Module):
    def __init__(
        self,
        dim,
        num_gates,
        eps = 1e-9,
        ##
        top_k = 1,
        outer_expert_dims = tuple(),
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 4.,
        capacity_factor_eval = 4.
        #capacity_factor_train = 1.25,
        #capacity_factor_eval = 2.
        ):
        super().__init__()

        self.eps = eps
        self.num_gates = num_gates
        self.top_k = top_k  
        # self.w_gating = nn.Parameter(torch.randn(*outer_expert_dims, dim, num_gates))
        self.w_gating = nn.Parameter(torch.randn(dim, num_gates))

        self.second_policy_train = second_policy_train
        self.second_policy_eval = second_policy_eval
        self.second_threshold_train = second_threshold_train
        self.second_threshold_eval = second_threshold_eval
        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval


        # for checking distribution
        self.last_indices = None
        self.last_masks = None
        self.last_gates = None
        self.last_raw_gates = None

    def forward(self, x, importance=None):
        T, B, dim, H, W = x.shape
        group_size = H * W
        num_gates = self.num_gates

        if self.training:
            policy = self.second_policy_train
            threshold = self.second_threshold_train
            capacity_factor = self.capacity_factor_train
        else:
            policy = self.second_policy_eval
            threshold = self.second_threshold_eval
            capacity_factor = self.capacity_factor_eval

        # Pool over time
        x_tok = x.flatten(3).permute(0, 1, 3, 2).contiguous()  # T, B, N, C
        x_pooled = x_tok.mean(dim=0)  # B, N, C

        raw_gates = torch.einsum('bnd,de->bne', x_pooled, self.w_gating)
        raw_gates = raw_gates.softmax(dim=-1)

        # Store masks, gates, indices, and positions for all top-k
        masks = []
        gates = []
        indices = []
        positions = []
        
        gates_remaining = raw_gates.clone()
        cumulative_mask_count = torch.zeros(B, 1, num_gates).to(raw_gates.device)  # Track capacity usage
        
        # Find top-k experts iteratively
        for k in range(self.top_k):
            gate_k, index_k = top1(gates_remaining)  # B, N
            mask_k = F.one_hot(index_k, num_gates).float()  # B, N, E
            
            # Apply importance filtering -- however, not used currently
            if importance is not None:
                if k == 0:
                    # First expert: only importance == 1
                    importance_mask = (importance == 1.).float()
                else:
                    # Other experts: importance > 0
                    importance_mask = (importance > 0.).float()
                
                mask_k *= importance_mask[..., None]
                gate_k *= importance_mask
            
            # Compute position in expert
            position_k = cumsum_exclusive(mask_k, dim=-2) + cumulative_mask_count
            position_k = position_k * mask_k
            
            # Update cumulative count
            cumulative_mask_count = cumulative_mask_count + mask_k.sum(dim=-2, keepdim=True)
            
            masks.append(mask_k)
            gates.append(gate_k)
            indices.append(index_k)
            positions.append(position_k.sum(dim=-1))  # B, N
            
            # Remove selected expert from remaining gates for next iteration
            gates_remaining = gates_remaining * (1. - mask_k)
        
        # Normalize gates
        gate_sum = sum(gates) + self.eps
        gates = [g / gate_sum for g in gates]
        
        # Compute balancing loss (using first expert)
        density_1 = masks[0].mean(dim=-2)
        density_1_proxy = raw_gates.mean(dim=-2)
        loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)
        
        # Apply policies to non-first experts
        for k in range(1, self.top_k):
            if policy == "all":
                pass
            elif policy == "none":
                masks[k] = torch.zeros_like(masks[k])
            elif policy == "threshold":
                masks[k] *= (gates[k] > threshold).float().unsqueeze(-1)
            elif policy == "random":
                probs = torch.zeros_like(gates[k]).uniform_(0., 1.)
                masks[k] *= (probs < (gates[k] / max(threshold, self.eps))).float().unsqueeze(-1)
        
        # Compute expert capacity
        expert_capacity = min(group_size, int((group_size * capacity_factor) / num_gates))
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)
        
        # Apply capacity constraints and recompute positions
        cumulative_position = torch.zeros(B, 1, num_gates).to(raw_gates.device)
        for k in range(self.top_k):
            position_k = cumsum_exclusive(masks[k], dim=-2) + cumulative_position
            position_k = position_k * masks[k]
            
            # Apply capacity constraint
            masks[k] *= (position_k < expert_capacity_f).float()
            
            # Update positions and gates based on capacity
            mask_k_flat = masks[k].sum(dim=-1)
            positions[k] = position_k.sum(dim=-1)
            gates[k] *= mask_k_flat
            
            # Update cumulative position
            cumulative_position = cumulative_position + masks[k].sum(dim=-2, keepdim=True)
        
        # Build combine and dispatch tensors
        combine_tensor = torch.zeros(B, group_size, num_gates, expert_capacity).to(raw_gates.device)
        
        for k in range(self.top_k):
            mask_k_flat = masks[k].sum(dim=-1)
            combine_tensor += (
                gates[k][..., None, None]
                * mask_k_flat[..., None, None]
                * F.one_hot(indices[k], num_gates)[..., None]
                * safe_one_hot(positions[k].long(), expert_capacity)[..., None, :]
            )
        
        dispatch_tensor = combine_tensor.bool().to(combine_tensor)

        # Cache latest routing info for inspection during inference
        self.last_indices = indices
        self.last_masks = masks
        self.last_gates = gates
        self.last_raw_gates = raw_gates
        
        return dispatch_tensor, combine_tensor, loss




class MoE(nn.Module):
    def __init__(self,
        dim,
        num_experts = 2,
        hidden_features=None,
        out_features=None,
        spike_mode="lif",
        second_policy_train = 'random',
        second_policy_eval = 'random',
        second_threshold_train = 0.2,
        second_threshold_eval = 0.2,
        capacity_factor_train = 1.25,
        capacity_factor_eval = 2.,
        loss_coef = 1e-2,

        ##
        top_k = 1,  
        experts = None):
        super().__init__()

        self.num_experts = num_experts
        tau_min = 2
        tau_max = 2

        gating_kwargs = {'top_k' : top_k, 'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
        # gating_kwargs = {'second_policy_train': second_policy_train, 'second_policy_eval': second_policy_eval, 'second_threshold_train': second_threshold_train, 'second_threshold_eval': second_threshold_eval, 'capacity_factor_train': capacity_factor_train, 'capacity_factor_eval': capacity_factor_eval}
        self.gate = Top2Gating(dim, num_gates = num_experts, **gating_kwargs)
        self.experts = nn.ModuleList([
            MS_MLP_Expert(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=out_features,
                spike_mode=spike_mode,
                tau=2.0,
            )
            for i in range(num_experts)
        ])
        self.loss_coef = loss_coef

    def forward(self, inputs, hook = None, **kwargs):
        ## steps for MoE
        ## 1. Gate based on input --> top k is received
        ## 2. token is made using input : T, B, N, D
        ## 3. expert_in = token input to each expert
        ## 4. for each expert, give expert inputs shaped T, B, Ccap, D. These are transformed into T, B, D, 1, Ccap to run MLP_expert(receives T B C H W shape input)
        #  Then retransformed to T, B, Ccap D
        ## 5. with the appended expert output, which is the output of each experts, we combine them with combine tensor
        ## 6. reshape it back to original data shape

        # b, n, d, e = *inputs.shape, self.num_experts
        T, B, D, H, W = inputs.shape
        N = H * W
        E = self.num_experts

        dispatch_tensor, combine_tensor, loss = self.gate(inputs)

        Ccap = combine_tensor.shape[-1]

        ## step 2
        x_tok = inputs.flatten(3).permute(0, 1, 3, 2).contiguous() # T, B, N, D

        expert_inputs = torch.einsum('tbnd,bnec->tebcd', x_tok, dispatch_tensor.to(x_tok.dtype))

        # Step 4: Process each expert
        expert_outputs_list = []
        
        for expert_id in range(E):
            # expert : MLP with channel projection (1x1) --> change input shape from T, B, Ccap, D to T, B, D, 1, Ccap
            expert_input = expert_inputs[:, expert_id, :, :, :]  # (T, B, Ccap, D)

            expert_input_spatial = expert_input.permute(0, 1, 3, 2) # T, B, D, Ccap
            expert_input_spatial = expert_input_spatial.unsqueeze(-2) # T, B, D, 1, Ccap

            expert_output_spatial, _ = self.experts[expert_id](expert_input_spatial, hook=hook)
            expert_output = expert_output_spatial.squeeze(-2)  # (T, B, D, Ccap)
            expert_output = expert_output.permute(0, 1, 3, 2)  # (T, B, Ccap, D)
            
            
            expert_outputs_list.append(expert_output)
        
        # Stack all expert outputs: (E, T, B, Ccap, D) → (T, E, B, Ccap, D)
        expert_outputs = torch.stack(expert_outputs_list, dim=1)
        
        # Step 5: Combine expert outputs using combine_tensor
        # (T, E, B, Ccap, D) x (B, N, E, Ccap) → (T, B, N, D)
        output = torch.einsum('tebcd,bnec->tbnd', expert_outputs, combine_tensor)
        
        # Reshape back to spatial format: (T, B, N, D) → (T, B, D, H, W)
        output = output.permute(0, 1, 3, 2).reshape(T, B, D, H, W)

        ###########################################
        ## add residual connection after experts ##
        ###########################################
        output = output + inputs

        return output, loss * self.loss_coef, hook
    
class MS_Block_Conv(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        attn_mode="direct_xor",
        spike_mode="lif",
        dvs=False,
        layer=0,
        num_experts = 2,
        loss_coef=1e-2,

        ##
        top_k = 1,  
    ):
        super().__init__()

        self.attn = MS_SSA_Conv(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            mode=attn_mode,
            spike_mode=spike_mode,
            dvs=dvs,
            layer=layer,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer = layer

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = MoE(
            dim=dim,
            num_experts=num_experts,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            spike_mode=spike_mode,
            loss_coef=loss_coef,

            ##
            top_k = top_k
        )

    def forward(self, x, hook=None):
        x_attn, attn, hook = self.attn(x, hook=hook)
        x, moe_loss, hook = self.mlp(x_attn, hook=hook)
        if hook is not None:
            hook[f"moe_loss_layer_{self.layer}"] = moe_loss
        return x, attn, hook