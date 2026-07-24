import torch
import torch.nn.functional as F

import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['Z', 'T']
)
@triton.jit
def gka_fwd_kernel(
    Q_2, K_2, V_2, G_2, Y_2, M_2, 
    stride_qz, stride_qt, stride_qu, stride_qv,
    stride_gz, stride_gt, stride_gu,
    Z, T, U: tl.constexpr, V: tl.constexpr,
):
    pid0 = tl.program_id(0)

    offset_u = tl.arange(0, U)
    offset_v = tl.arange(0, V)

    acc1 = tl.zeros([V, V], dtype=tl.float32)
    acc2 = tl.zeros([V,], dtype=tl.float32)
    q_like_offset = pid0 * stride_qz + offset_u[:,None] * stride_qu + offset_v[None,:] * stride_qv
    g_like_offset = pid0 * stride_gz + offset_u * stride_gu

    for t in range(T):
        g_2_ptrs = G_2 + g_like_offset
        g_2_ori = tl.load(g_2_ptrs)
        g_2 = tl.cumsum(g_2_ori, 0)
        g_diff_2 = g_2[None,:] - g_2[:,None]
        g_diff_clamp_2 = tl.minimum(g_diff_2, 1.0)
        f_2 = tl.exp(g_diff_clamp_2)
        f_2 = f_2.to(Q_2.dtype.element_ty)

        q_2_ptrs = Q_2 + q_like_offset
        q_2_ori = tl.load(q_2_ptrs)
        q_2 = tl.where(q_2_ori > 0, q_2_ori*q_2_ori, 0)
        
        k_2_ptrs = K_2 + q_like_offset
        k_2_ori = tl.load(k_2_ptrs)
        k_2 = tl.where(k_2_ori > 0, k_2_ori*k_2_ori, 0)

        w_2 = tl.dot(q_2, tl.trans(k_2))
        n_2 = w_2 * f_2
        mask = offset_u[:, None] >= offset_u[None, :]
        n_2 = tl.where(mask, n_2, 0.).to(Q_2.dtype.element_ty)
        m_2 = tl.sum(n_2, 1, True) + 0.001
        v_2_ptrs = V_2 + q_like_offset
        v_2 = tl.load(v_2_ptrs)
        a_2 = tl.dot(n_2, v_2)
        f_ik_2 = tl.exp(-g_2).to(Q_2.dtype.element_ty)
        qf_2 = q_2 * f_ik_2[:, None]
        a_2 += tl.dot(qf_2, acc1.to(Q_2.dtype.element_ty))
        m_2 += tl.dot(qf_2, acc2[:, None].to(Q_2.dtype.element_ty))
        y_2 = a_2 / m_2
        m_2_ptrs = M_2 + g_like_offset
        tl.store(m_2_ptrs, tl.sum(m_2, 1))
        y_2_ptrs = Y_2 + q_like_offset
        tl.store(y_2_ptrs, y_2)

        last = tl.sum(g_2_ori, 0)
        c = tl.exp(-last).to(Q_2.dtype.element_ty)
        acc1 = acc1 * c
        acc2 = acc2 * c
        f_kj_2 = tl.exp(g_2 - last).to(Q_2.dtype.element_ty)
        fk_2 = k_2 * f_kj_2[:,None]
        fkv_2 = tl.dot(tl.trans(fk_2), v_2)
        acc1 += fkv_2
        fks_2 = tl.sum(fk_2, 0)
        acc2 += fks_2
        q_like_offset += stride_qt
        g_like_offset += stride_gt

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['Z', 'T']
)
@triton.jit
def gka_bwd_kernel(
    Q_2, K_2, V_2, G_2, Y_2, M_2, 
    dQ_2, dK_2, dV_2, dG_2, dA_2, dM_2,
    stride_qz, stride_qt, stride_qu, stride_qv,
    stride_gz, stride_gt, stride_gu,
    Z, T, U: tl.constexpr, V: tl.constexpr,
):
    pid0 = tl.program_id(0)

    offset_u = tl.arange(0, U)
    offset_v = tl.arange(0, V)

    mask = offset_u[:, None] >= offset_u[None, :]
    mask2 = offset_u[:, None] > offset_u[None, :]

    q_like_offset = pid0 * stride_qz + offset_u[:,None] * stride_qu + offset_v[None,:] * stride_qv
    g_like_offset = pid0 * stride_gz + offset_u * stride_gu

    acc1 = tl.zeros([V, V], dtype=tl.float32)
    acc2 = tl.zeros([V,], dtype=tl.float32)

    for t in range(T):

        g_2_ptrs = G_2 + g_like_offset
        g_2_ori = tl.load(g_2_ptrs)
        g_2 = tl.cumsum(g_2_ori, 0)
        
        g_diff_2 = g_2[None,:] - g_2[:,None]
        g_diff_clamp_2 = tl.minimum(g_diff_2, 1.0)
        f_2 = tl.exp(g_diff_clamp_2)
        f_2 = f_2.to(Q_2.dtype.element_ty)

        q_2_ptrs = Q_2 + q_like_offset
        q_2_ori = tl.load(q_2_ptrs)
        q_2 = tl.where(q_2_ori > 0, q_2_ori*q_2_ori, 0)
        
        k_2_ptrs = K_2 + q_like_offset
        k_2_ori = tl.load(k_2_ptrs)
        k_2 = tl.where(k_2_ori > 0, k_2_ori*k_2_ori, 0)

        w_2 = tl.dot(q_2, tl.trans(k_2))
        n_2_full = w_2 * f_2
        n_2 = tl.where(mask, n_2_full, 0.).to(Q_2.dtype.element_ty)
        da_2_ptrs = dA_2 + q_like_offset
        da_2 = tl.load(da_2_ptrs)
        v_2_ptrs = V_2 + q_like_offset
        v_2 = tl.load(v_2_ptrs)
        dav_2 = tl.dot(da_2, tl.trans(v_2))
        dm_2_ptrs = dM_2 + g_like_offset
        dm_2 = tl.load(dm_2_ptrs)
        dn_2_full = dav_2 + dm_2[:, None]
        dn_2 = tl.where(mask, dn_2_full, 0.).to(Q_2.dtype.element_ty)
        dnf_2 = dn_2 * f_2
        dq_2_diag = tl.dot(dnf_2, k_2)
        u_2_full = dn_2 * n_2
        u_2 = tl.where(mask2, u_2_full, 0.)
        dg_2_diag = tl.sum(u_2, 0) - tl.sum(u_2, 1)
        dg_2_diag = dg_2_diag.to(tl.float32)

        f_ik_2 = tl.exp(-g_2).to(Q_2.dtype.element_ty)
        daf_2 = da_2 * f_ik_2[:, None]
        dq_2_part = tl.dot(daf_2, acc1.to(Q_2.dtype.element_ty))
        dmf_2 = dm_2 * f_ik_2
        dq_2_part += dmf_2[:, None] * acc2.to(Q_2.dtype.element_ty)[None,:]
        dq_2_ptrs = dQ_2 + q_like_offset

        dq_2 = dq_2_part + dq_2_diag
        dq_2 = tl.where(q_2_ori > 0, dq_2 * q_2_ori * 2, 0)
        tl.store(dq_2_ptrs, dq_2)
        
        dg_2_part = tl.sum(dq_2_part * q_2, 1, dtype=tl.float32)
        dg_2_ptrs = dG_2 + g_like_offset
        tl.store(dg_2_ptrs, dg_2_diag-dg_2_part)

        last = tl.sum(g_2_ori, 0)
        c = tl.exp(-last).to(Q_2.dtype.element_ty)
        acc1 = acc1 * c
        acc2 = acc2 * c
        f_kj_2 = tl.exp(g_2 - last).to(Q_2.dtype.element_ty)
        fk_2 = k_2 * f_kj_2[:,None]
        vfk_2 = tl.dot(tl.trans(v_2), fk_2)
        acc1 += vfk_2
        fks_2 = tl.sum(fk_2, 0)
        acc2 += fks_2

        q_like_offset += stride_qt
        g_like_offset += stride_gt

    acc1 = tl.zeros([V, V], dtype=tl.float32)
    acc2 = tl.zeros([V,], dtype=tl.float32)
    
    dg_2_offset = 0.

    for t in range(T-1,-1,-1):
        q_like_offset -= stride_qt
        g_like_offset -= stride_gt

        q_2_ptrs = Q_2 + q_like_offset
        q_2_ori = tl.load(q_2_ptrs)
        q_2 = tl.where(q_2_ori > 0, q_2_ori*q_2_ori, 0)
        
        k_2_ptrs = K_2 + q_like_offset
        k_2_ori = tl.load(k_2_ptrs)
        k_2 = tl.where(k_2_ori > 0, k_2_ori*k_2_ori, 0)

        v_2_ptrs = V_2 + q_like_offset
        v_2 = tl.load(v_2_ptrs)
        g_2_ptrs = G_2 + g_like_offset
        g_2_ori = tl.load(g_2_ptrs)
        g_2 = tl.cumsum(g_2_ori, 0)
        
        g_diff_2 = g_2[None,:] - g_2[:,None]
        g_diff_clamp_2 = tl.minimum(g_diff_2, 1.0)
        f_2 = tl.exp(g_diff_clamp_2)
        f_2 = f_2.to(Q_2.dtype.element_ty)
        w_2 = tl.dot(q_2, tl.trans(k_2))
        n_2_full = w_2 * f_2
        n_2 = tl.where(mask, n_2_full, 0.).to(Q_2.dtype.element_ty)
        da_2_ptrs = dA_2 + q_like_offset
        da_2 = tl.load(da_2_ptrs)
        dv_2_diag = tl.dot(tl.trans(n_2), da_2)
        dav_2 = tl.dot(da_2, tl.trans(v_2))
        dm_2_ptrs = dM_2 + g_like_offset
        dm_2 = tl.load(dm_2_ptrs)
        dn_2_full = dav_2 + dm_2[:, None]
        dn_2 = tl.where(mask, dn_2_full, 0.).to(Q_2.dtype.element_ty)
        dnf_2 = dn_2 * f_2
        dk_2_diag = tl.dot(tl.trans(dnf_2), q_2)
        
        tot = tl.sum(g_2_ori, 0)
        f_kj_2 = tl.exp(g_2 - tot).to(Q_2.dtype.element_ty)
        dvf_2 = v_2 * f_kj_2[:, None]
        dk_2_part = tl.dot(dvf_2, acc1.to(Q_2.dtype.element_ty))
        dk_2_part += f_kj_2[:, None] * acc2.to(Q_2.dtype.element_ty)[None, :]
        dk_2_ptrs = dK_2 + q_like_offset
        
        dk_2 = dk_2_part + dk_2_diag
        dk_2 = tl.where(k_2_ori > 0, dk_2 * k_2_ori * 2, 0)
        tl.store(dk_2_ptrs, dk_2)

        dg_2_part = tl.sum(dk_2_part * k_2, 1, dtype=tl.float32)
        dg_2_ptrs = dG_2 + g_like_offset
        dg_2_prv = tl.load(dg_2_ptrs)
        dg_2_ori = dg_2_prv + dg_2_part
        dg_2 = tl.cumsum(dg_2_ori, 0, reverse=True) + dg_2_offset
        dg_2_offset = dg_2_offset + tl.sum(dg_2_ori, 0)
        tl.store(dg_2_ptrs, dg_2)
        dv_2_part = tl.dot(k_2 * f_kj_2[:, None], tl.trans(acc1).to(Q_2.dtype.element_ty))
        dv_2_ptrs = dV_2 + q_like_offset
        tl.store(dv_2_ptrs, dv_2_part + dv_2_diag)

        c = tl.exp(-tot).to(Q_2.dtype.element_ty)
        acc1 = acc1 * c
        acc2 = acc2 * c
        f_ik_2 = tl.exp(-g_2).to(Q_2.dtype.element_ty)
        fq_2 = q_2 * f_ik_2[:,None]
        dafq_2 = tl.dot(tl.trans(da_2), fq_2)
        acc1 += dafq_2
        fsq_2 = tl.sum(dm_2[:,None]*fq_2, 0)
        acc2 += fsq_2

class GatedKernelAttention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_: torch.Tensor, k_: torch.Tensor, v_: torch.Tensor, g_: torch.Tensor) -> torch.Tensor:
        U = 32
        B, H, L, V = q_.shape
        T = ((L-1)//U + 1)
        extra = T * U - L
        Z = B * H

        if extra:
            q_ = F.pad(q_, (0, 0, 0, extra), mode='constant')
            k_ = F.pad(k_, (0, 0, 0, extra), mode='constant')
            v_ = F.pad(v_, (0, 0, 0, extra), mode='constant')
            g_ = F.pad(g_, (0, 0, 0, extra), mode='constant')

        q_2 = q_.reshape(Z, T, U, V).contiguous()
        k_2 = k_.reshape(Z, T, U, V).contiguous()
        v_2 = v_.reshape(Z, T, U, V).contiguous()
        g_2 = g_.reshape(Z, T, U).contiguous()

        y_2 = torch.zeros(Z, T, U, V, device=q_.device, dtype=q_.dtype)
        m_2 = torch.zeros(Z, T, U, 1, device=q_.device, dtype=q_.dtype)

        grid = (Z, 1)

        gka_fwd_kernel[grid](
            q_2, k_2, v_2, g_2, y_2, m_2,
            q_2.stride(0), q_2.stride(1), q_2.stride(2), q_2.stride(3),
            g_2.stride(0), g_2.stride(1), g_2.stride(2),
            Z, T, U, V,
        )
        ctx.save_for_backward(q_2, k_2, v_2, g_2, y_2, m_2)
        ctx.shapes = (B, H, L, T, U, V, Z, extra)
        y_ = y_2.reshape(B, H, T*U, V)
        if extra:
            y_ = y_[:,:,:L,:]
        return y_

    @staticmethod
    def backward(ctx, dy_: torch.Tensor):
        q_2, k_2, v_2, g_2, y_2, m_2 = ctx.saved_tensors
        B, H, L, T, U, V, Z,extra = ctx.shapes
        dtype = q_2.dtype

        if extra:
            dy_ = F.pad(dy_, (0, 0, 0, extra), mode='constant')

        dy_2 = dy_.reshape_as(y_2)
        da_2 = dy_2 / m_2
        dm_2 = -torch.sum(da_2 * y_2, -1, True, dtype=torch.float32).to(dtype)
        da_2 = da_2.contiguous()
        dm_2 = dm_2.contiguous()

        dq_2 = torch.zeros_like(q_2)
        dk_2 = torch.zeros_like(k_2)
        dv_2 = torch.zeros_like(v_2)
        dg_2 = torch.zeros_like(g_2)

        grid = (Z, 1)

        gka_bwd_kernel[grid](
            q_2, k_2, v_2, g_2, y_2, m_2,
            dq_2, dk_2, dv_2, dg_2, da_2, dm_2,
            q_2.stride(0), q_2.stride(1), q_2.stride(2), q_2.stride(3),
            g_2.stride(0), g_2.stride(1), g_2.stride(2),
            Z, T, U, V,
        )

        dq_ = dq_2.reshape(B, H, T*U, V)
        dk_ = dk_2.reshape(B, H, T*U, V)
        dv_ = dv_2.reshape(B, H, T*U, V)
        dg_ = dg_2.reshape(B, H, T*U, 1)
        
        if extra:
            dq_ = dq_[:,:,:L,:]
            dk_ = dk_[:,:,:L,:]
            dv_ = dv_[:,:,:L,:]
            dg_ = dg_[:,:,:L,:]

        return dq_, dk_, dv_, dg_
    
def gated_kernel_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    return GatedKernelAttention.apply(q, k, v, g)

def gated_kernel_attention_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    q_ = q.relu().square()
    k_ = k.relu().square()
    v_ = v
    g_ = g.float()

    f1 = torch.cumsum(g_, dim=2)
    f2 = f1.transpose(2, 3)
    factor = torch.exp((f2 - f1).clamp_max(10))
    dot = q_ @ k_.transpose(-2, -1)
    num = torch.tril(dot * factor).to(q.dtype)
    den = num.sum(dim=-1, keepdim=True).add(1e-3)
    y = (num @ v_) / den
    y = y.transpose(1, 2) # (B, T, H, D)
    return y

def gated_kernel_attention_recurrent(
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        g: torch.Tensor, 
        num_cache: torch.Tensor, 
        den_cache: torch.Tensor,
) -> torch.Tensor:
    '''
    Implementation for recurrent version of custom attention mechanism
    q.shape == (B, 1, H, D)
    k.shape == (B, 1, H, D)
    v.shape == (B, 1, H, D)
    g.shape == (B, 1, H, 1)
    num_cache.shape == (B, H, D, D)
    den_cache.shape == (B, H, D, 1)
    '''

    q_ = q.relu().square().float()
    k_ = k.relu().square().float()
    v_ = v.float()
    g_ = g.float()

    factor = g_.neg().exp()
    k_T = k_.transpose(-2, -1) # (B, H, D, 1)
    num_cache = (num_cache * factor + (k_T @ v_))
    den_cache = (den_cache * factor + (k_T))
    y = (q_ @ num_cache) / (q_ @ den_cache).add(0.001) # (B, H, 1, D)
    return y, num_cache, den_cache