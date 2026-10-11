import torch
from torch import autograd 

class FlashAttentionFunction(autograd.Function):
    @staticmethod
    def forward(ctx, 
                q: torch.Tensor, 
                k: torch.Tensor, 
                v: torch.Tensor, 
                is_causal: bool =False) -> torch.Tensor:
        batch_size = q.shape[0]
        q_seq_len = q.shape[1]
        n_dim = q.shape[2]
        kv_seq_len = k.shape[1] #注意这里

        q_tile_size = 16
        q_rows = q_seq_len // q_tile_size 
        kv_tile_size = 16
        kv_rows = kv_seq_len // kv_tile_size
        o = torch.zeros_like(q) 
        l = torch.zeros([batch_size, q_seq_len])

        for batch_idx in range(batch_size):
            for i in range(q_rows):
                q_i = q[batch_idx, i * q_tile_size : (i + 1) * q_tile_size] # [q_tile_size, n_dim]
                o_i = torch.zeros_like(q_i)   # [q_tile_size, n_dim]
                l_i = torch.zeros(q_tile_size)    # [q_tile_size]
                m_i_last = torch.tensor([-1e9] * q_tile_size)    # [q_tile_size]
                for j in range(kv_rows):
                    k_j = k[batch_idx, j * kv_tile_size : (j + 1) * kv_tile_size]  # [kv_tile_size, n_dim]
                    v_j = v[batch_idx, j * kv_tile_size : (j + 1) * kv_tile_size]  # [kv_tile_size, n_dim]
                    q_scale = n_dim ** 0.5 
                    s_ij = q_i @ k_j.T / q_scale  # [q_tile_size, kv_tile_size] 不能用* 必须用@乘号
                    m_ij = torch.max(s_ij, dim = -1, keepdim=False)[0] # [q_tile_size]
                    m_i = torch.max(torch.stack([m_i_last, m_ij], dim = -1), dim=-1, keepdim=False)[0] # [q_tile_size] max返回结果需要返回元组的第一个
                    p_ij = torch.exp(s_ij - m_i[:, None]) #  [q_tile_size, kv_tile_size] softmax的分子 注意这里如果不给m_i扩展维度，会维度不匹配，因为不满足广播条件
                    l_i = l_i * torch.exp(m_i_last - m_i) + torch.sum(p_ij, dim=-1, keepdim=False) # [q_tile_size] softmax分母的部分和 注意这里可以用*乘号
                    o_i = torch.diag(torch.exp(m_i_last - m_i)) @ o_i + p_ij @ v_j # [q_tile_size, n_dim]
                    m_i_last = m_i  
                o[batch_idx, i * q_tile_size : (i + 1) * q_tile_size] = torch.diag(1.0 / l_i) @ o_i # [q_tile_size, n_dim]
                l[batch_idx, i * q_tile_size : (i + 1) * q_tile_size] = m_i_last + torch.log(l_i) # [q_tile_size]

        ctx.save_for_backward(q, k, v, o, l)
        ctx.is_causal = is_causal
        
        return o      