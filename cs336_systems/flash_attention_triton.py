import torch
from torch import autograd 
import triton 
from triton import language as tl 

@triton.jit 
def flash_fwd_kernel(
    q_ptr, k_ptr, v_ptr,
    o_ptr, l_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    q_seq_len, kv_seq_len,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_casual: tl.constexpr
    ):
    # Program indices
    q_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
 
    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    q_block_ptr = tl.make_block_ptr(
        q_ptr + batch_index * stride_qb,
        shape=(q_seq_len, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    ) 

    k_block_ptr = tl.make_block_ptr(
        k_ptr + batch_index * stride_kb, 
        shape=(kv_seq_len, D), 
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)   
    ) 

    v_block_ptr = tl.make_block_ptr(
        v_ptr + batch_index * stride_vb, 
        shape=(kv_seq_len, D), 
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1,0)   
    )

    o_block_ptr =  tl.make_block_ptr(
        o_ptr + batch_index * stride_ob, 
        shape=(q_seq_len, D), 
        strides=(stride_oq, stride_od),
        offsets=(q_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1,0)   
    ) 

    l_block_ptr = tl.make_block_ptr(
        l_ptr + batch_index * stride_lb, 
        shape=(q_seq_len,), 
        strides=(stride_lq, ), #这个必须是元组
        offsets=(q_tile_index * Q_TILE_SIZE, ),
        block_shape=(Q_TILE_SIZE, ),
        order=(0,)   
    ) 
    q_i = tl.load(q_block_ptr) 
    m_i = tl.full([Q_TILE_SIZE], -1e9, dtype=tl.float32) # 1e-9
    l_i = tl.load(l_block_ptr) 
    o_i = tl.load(o_block_ptr) 
    for j in range(tl.cdiv(kv_seq_len, K_TILE_SIZE)):
        k_j = tl.load(k_block_ptr) 
        v_j = tl.load(v_block_ptr) 
        s_ij = tl.dot(q_i, tl.trans(k_j)) / scale 
        if is_casual:
            q_index = q_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)  # tl.arange参数必须传入tl.constexpr
            k_index = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE) 
            mask = q_index[:, None] < k_index[None, :] 
            s_ij = tl.where(mask, -1e9, s_ij)
        m_ij = tl.max(s_ij, axis = -1, keep_dims=False) 
        m_i_new = tl.maximum(m_i, m_ij) 
        p_ij = tl.exp(s_ij - tl.expand_dims(m_i_new, 1)) 
        l_i = l_i * tl.exp(m_i - m_i_new) + tl.sum(p_ij, axis=1) 
        o_i = tl.expand_dims(tl.exp(m_i - m_i_new), 1) * o_i + tl.dot(p_ij.to(v_j.dtype),v_j) 
        m_i = m_i_new 

        k_block_ptr = tl.advance(k_block_ptr, [K_TILE_SIZE, 0]) 
        v_block_ptr = tl.advance(v_block_ptr, [K_TILE_SIZE, 0])
    o_i = o_i / tl.expand_dims(l_i, 1) 
    l_i = m_i + tl.log(l_i) 

    tl.store(o_block_ptr, o_i) 
    tl.store(l_block_ptr, l_i) 


class FlashAttentionTriton(autograd.Function):
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
        o = torch.zeros_like(q, device=q.device)  #必须在gpu上
        l = torch.zeros([batch_size, q_seq_len], dtype=q.dtype, device=q.device)

        grid = [q_rows, batch_size]
        flash_fwd_kernel[grid](
            q, k, v, 
            o, l,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            l.stride(0), l.stride(1), 
            q_seq_len, kv_seq_len, 
            n_dim ** 0.5, 
            n_dim, 
            q_tile_size, 
            kv_tile_size,
            is_causal
        )
        
        ctx.save_for_backward(q, k, v, o, l)
        ctx.is_causal = is_causal
        
        return o      