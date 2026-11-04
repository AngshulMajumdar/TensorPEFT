# Copyright 2026 Angshul Majumdar.
"""Rank-one CP approximation by higher-order power iteration (HOPM)."""
import torch

_L = "abcdefgh"


def contract_except(T, us, n):
    N = T.dim()
    idx = _L[:N]
    spec = idx + "," + ",".join(idx[m] for m in range(N) if m != n) + "->" + idx[n]
    return torch.einsum(spec, T, *[us[m] for m in range(N) if m != n])


def cp_value(T, us):
    idx = _L[: T.dim()]
    return torch.einsum(idx + "," + ",".join(idx) + "->", T, *us)


def outer(us):
    out = us[0]
    for u in us[1:]:
        out = out[..., None] * u
    return out


@torch.no_grad()
def rank1_cp(T, iters=30, tol=1e-6):
    """Unit factors u_1..u_N (approximately) maximising sigma = <T, u_1 o ... o u_N>.

    T has shape (leading..., d_in, d_out) with one or two leading modes.  Initialised from the top
    singular pair of the largest (d_in x d_out) slice (deterministic), then HOPM sweeps."""
    N = T.dim()
    lead_shape = T.shape[:-2]
    slices = T.reshape(-1, T.shape[-2], T.shape[-1])
    j = int(slices.flatten(1).norm(dim=1).argmax())
    M = slices[j]
    small = M @ M.T if M.shape[0] <= M.shape[1] else M.T @ M        # exact top pair via Gram eigvec
    _, vecs = torch.linalg.eigh(small.double())
    v = vecs[:, -1].to(T.dtype)
    if M.shape[0] <= M.shape[1]:
        u_in, u_out = v, M.T @ v
    else:
        u_in, u_out = M @ v, v
    u_in, u_out = u_in / u_in.norm().clamp_min(1e-30), u_out / u_out.norm().clamp_min(1e-30)
    core = torch.einsum("sij,i,j->s", slices, u_in, u_out).reshape(lead_shape)
    if core.dim() == 1:
        us = [core / core.norm().clamp_min(1e-30)]
    else:
        Uc, _, Vc = torch.linalg.svd(core, full_matrices=False)
        us = [Uc[:, 0], Vc[0]]
    us = us + [u_in, u_out]
    sigma = cp_value(T, us)
    for _ in range(iters):
        for n in range(N):
            w = contract_except(T, us, n)
            us[n] = w / w.norm().clamp_min(1e-30)
        new = cp_value(T, us)
        done = abs(float(new) - float(sigma)) <= tol * max(abs(float(new)), 1e-30)
        sigma = new
        if done:
            break
    if sigma < 0:
        us[-1] = -us[-1]
        sigma = -sigma
    return us, float(sigma)
