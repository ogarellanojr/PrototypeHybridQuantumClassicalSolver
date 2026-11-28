import numpy as np
from scipy.linalg import expm
from numpy.polynomial.chebyshev import chebval

# ==============================================================================
# 1. Exact QSVT Solver for 1-Year Earthquake Distribution (Steps 1-4)
# ==============================================================================
def solve_earthquake_qsvt(m_eq=1.0, N_t=16):
    # Table 2 implied compound Poisson rates (per year)
    lam_1 = np.array([0.016147, 0.005498, 0.006215, 0.011743])
    lam_2 = np.array([0.022711, 0.004020, 0.002915, 0.010395])
    Lam_1, Lam_2 = np.sum(lam_1), np.sum(lam_2)
    w_1, w_2 = lam_1 / Lam_1, lam_2 / Lam_2

    # Shared-event rate c (Eq. 4)
    c = m_eq * 0.5 * (0.075 * Lam_1 + 0.074 * Lam_2)

    # 25x25 CTMC generator matrix G over (a, b) in {0..4}^2
    G = np.zeros((25, 25), dtype=float)
    for a in range(5):
        for b in range(5):
            u = 5 * a + b
            for k in range(1, 5):
                v = 5 * min(a + k, 4) + b
                if v != u:
                    G[u, v] += (Lam_1 - c) * w_1[k - 1]
            for j in range(1, 5):
                v = 5 * a + min(b + j, 4)
                if v != u:
                    G[u, v] += (Lam_2 - c) * w_2[j - 1]
            for k in range(1, 5):
                for j in range(1, 5):
                    v = 5 * min(a + k, 4) + min(b + j, 4)
                    if v != u:
                        G[u, v] += c * w_1[k - 1] * w_2[j - 1]
            G[u, u] = -np.sum(G[u, :])

    # Space-time Euler discretization A for N_t steps
    dt = 1.0 / N_t
    M = np.eye(25) + dt * G.T
    dim_A = 25 * N_t
    A = np.eye(dim_A)
    for t in range(1, N_t):
        A[t * 25:(t + 1) * 25, (t - 1) * 25:t * 25] = -M

    b_vec = np.zeros(dim_A)
    b_vec[0] = 1.0

    # Hermitian dilation A_tilde and LCU walk operators U_±
    Z_A = np.zeros((dim_A, dim_A))
    A_tilde = np.block([[Z_A, A], [A.T.conj(), Z_A]])
    alpha = np.linalg.norm(A_tilde, ord=2)
    A_norm = A_tilde / alpha

    eigvals, eigvecs = np.linalg.eigh(A_norm)
    kappa = np.max(np.abs(eigvals)) / np.min(np.abs(eigvals))
    s_min = 1.0 / kappa
    scale = 0.5 * s_min
    degree = int(np.ceil(8 * kappa)) | 1

    xs = np.cos(np.pi * (2 * np.arange(1, 3001) - 1) / 6000)
    mask = np.abs(xs) >= s_min
    odd_degrees = np.arange(1, degree + 1, 2)
    T_mat = np.column_stack([np.cos(k * np.arccos(xs[mask])) for k in odd_degrees])
    weights = np.abs(xs[mask])
    c_odd, _, _, _ = np.linalg.lstsq(
        T_mat * weights[:, None], (scale / xs[mask]) * weights, rcond=None
    )

    cheb_coeffs = np.zeros(degree + 1)
    cheb_coeffs[odd_degrees] = c_odd
    max_poly = np.max(np.abs(chebval(xs, cheb_coeffs)))
    if max_poly > 0.95:
        rescale = 0.95 / max_poly
        cheb_coeffs *= rescale
        scale *= rescale

    P_eigs = chebval(eigvals, cheb_coeffs)
    P_A_tilde = eigvecs @ np.diag(P_eigs) @ eigvecs.T.conj()

    b_dil = np.concatenate([b_vec, np.zeros(dim_A)])
    out_dil = (P_A_tilde @ b_dil) * (alpha / scale)
    p_eq_qsvt = np.real(out_dil[dim_A:])[-25:]
    p_eq_qsvt = np.maximum(p_eq_qsvt, 0.0)
    p_eq_qsvt /= np.sum(p_eq_qsvt)  # Normalize distribution

    p_eq_exact = expm(G)[0, :]
    return p_eq_qsvt, p_eq_exact


# ==============================================================================
# 2. Precipitation Model & 100-State Annual Distribution (Steps 5-6)
# ==============================================================================
def build_annual_distribution(p_eq, rain_mult=1.0, rain_shared_share=0.25):
    q = 0.02
    mu = -np.log(1.0 - q) * rain_mult
    mu_1, mu_2 = mu, mu
    c_r = rain_shared_share * min(mu_1, mu_2)

    # Eq. 6: (r1, r2) probabilities
    P_00 = np.exp(-(mu_1 + mu_2 - c_r))
    P_01 = np.exp(-mu_1) - P_00
    P_10 = np.exp(-mu_2) - P_00
    P_11 = 1.0 - P_00 - P_10 - P_01
    p_rain = np.array([[P_00, P_01], [P_10, P_11]])

    # Annual state (a, b, r1, r2) -> shape (5, 5, 2, 2) = 100 states
    p_eq_grid = p_eq.reshape(5, 5)
    p_annual = p_eq_grid[:, :, None, None] * p_rain[None, None, :, :]
    return p_annual


# ==============================================================================
# 3. Three-Year Term Convolution & Focus Targets (Steps 7-8, Table 3)
# ==============================================================================
def evaluate_3yr_targets(p_annual, eq_limit=70e6, rain_limit=10e6, T_attach=80e6):
    # Direct convolution over 3 years with state capping at (4, 4, 1, 1)
    p_term = np.zeros((5, 5, 2, 2), dtype=float)
    p_term[0, 0, 0, 0] = 1.0

    for _ in range(3):
        p_next = np.zeros((5, 5, 2, 2), dtype=float)
        for a1 in range(5):
            for b1 in range(5):
                for r1_1 in range(2):
                    for r2_1 in range(2):
                        prob1 = p_term[a1, b1, r1_1, r2_1]
                        if prob1 == 0:
                            continue
                        for a2 in range(5):
                            for b2 in range(5):
                                for r1_2 in range(2):
                                    for r2_2 in range(2):
                                        prob2 = p_annual[a2, b2, r1_2, r2_2]
                                        na = min(a1 + a2, 4)
                                        nb = min(b1 + b2, 4)
                                        nr1 = min(r1_1 + r1_2, 1)
                                        nr2 = min(r2_1 + r2_2, 1)
                                        p_next[na, nb, nr1, nr2] += prob1 * prob2
        p_term = p_next

    # Map each term state to principal losses L1, L2, L (Eq. 7)
    a_idx, b_idx, r1_idx, r2_idx = np.meshgrid(
        np.arange(5), np.arange(5), np.arange(2), np.arange(2), indexing="ij"
    )
    L1 = eq_limit * (a_idx / 4.0) + rain_limit * r1_idx
    L2 = eq_limit * (b_idx / 4.0) + rain_limit * r2_idx
    L = L1 + L2
    bond_principal = eq_limit + rain_limit  # USD 80m

    # Table 3 Focus Targets
    el_usd = np.sum(p_term * L)
    joint_attach_prob = np.sum(p_term[(L1 > 0) & (L2 > 0)])
    exceed_80m_prob = np.sum(p_term[L > T_attach])
    stop_loss_usd = np.sum(p_term * np.maximum(L - T_attach, 0.0))
    both_exhausted_prob = np.sum(
        p_term[np.isclose(L1, bond_principal) & np.isclose(L2, bond_principal)]
    )

    return {
        "EL_m": el_usd / 1e6,
        "Both_pay_pct": joint_attach_prob * 100.0,
        "Exceed_80m_pct": exceed_80m_prob * 100.0,
        "Stop_loss_usd": stop_loss_usd,
        "Both_exhausted_pct": both_exhausted_prob * 100.0,
    }


# ==============================================================================
# 4. Run All 5 Benchmark Scenarios (Table 4 & Table 5)
# ==============================================================================
scenarios = [
    ("adb-base",         dict(m_eq=1.0, rain_mult=1.0, rain_shared_share=0.25, eq_limit=70e6, rain_limit=10e6)),
    ("adb-independent",  dict(m_eq=0.0, rain_mult=1.0, rain_shared_share=0.00, eq_limit=70e6, rain_limit=10e6)),
    ("adb-common-shock", dict(m_eq=2.0, rain_mult=1.0, rain_shared_share=0.25, eq_limit=70e6, rain_limit=10e6)),
    ("adb-rain-stress",  dict(m_eq=1.0, rain_mult=2.0, rain_shared_share=0.25, eq_limit=70e6, rain_limit=10e6)),
    ("adb-rain-cover",   dict(m_eq=1.0, rain_mult=1.0, rain_shared_share=0.25, eq_limit=60e6, rain_limit=20e6)),
]

header = f"{'Scenario':<18} | {'Method':<6} | {'Exp Loss':<9} | {'Both Pay':<8} | {'L > 80m':<8} | {'Stop Loss':<10} | {'Exhausted':<9}"
print(header)
print("-" * len(header))

for name, cfg in scenarios:
    p_eq_qsvt, p_eq_exact = solve_earthquake_qsvt(m_eq=cfg["m_eq"], N_t=16)

    for label, p_eq in [("Exact", p_eq_exact), ("Prototype", p_eq_qsvt)]:
        p_ann = build_annual_distribution(p_eq, cfg["rain_mult"], cfg["rain_shared_share"])
        res = evaluate_3yr_targets(p_ann, cfg["eq_limit"], cfg["rain_limit"])
        print(
            f"{name:<18} | {label:<6} | "
            f"${res['EL_m']:>6.2f}m | "
            f"{res['Both_pay_pct']:>6.3f}% | "
            f"{res['Exceed_80m_pct']:>6.3f}% | "
            f"${res['Stop_loss_usd']:>8,.0f} | "
            f"{res['Both_exhausted_pct']:>7.4f}%"
        )
