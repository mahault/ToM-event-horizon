"""
Externalization Horizon — Model-Similarity AIF Simulation Suite
================================================================
Principled active inference agents where:
1. Agent A does VFE minimization (KF perception) + EFE-optimal action (a = K*mu)
2. Observer B does sophisticated inference (KF-based ToM) over A's beliefs
3. Permeability EMERGES from model similarity: B can infer A's beliefs iff
   B's generative model matches A's actual dynamics
4. The externalization horizon = boundary in MODEL SPACE where B's inference fails

Key insight: the horizon is OBSERVER-DEPENDENT. Same agent A can be transparent to
one observer (matched model) and dark to another (mismatched model).

Authors: Mahault Albarracin, Alejandro Jimenez Rodriguez
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Publication-quality style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# ═══════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════

def steady_state_kalman(phi, sigma_x, sigma_o):
    """Steady-state Kalman gain for AR(1) state: x_{t+1} = phi*x_t + noise."""
    a = 1.0
    b = sigma_o**2 * (1 - phi**2) - sigma_x**2
    c = -sigma_x**2 * sigma_o**2
    P_pred = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
    K_f = P_pred / (P_pred + sigma_o**2)
    P_post = (1 - K_f) * P_pred
    return K_f, P_pred, P_post


def belief_variance(phi, sigma_x, sigma_o):
    """Steady-state temporal variance of Kalman-filtered beliefs Var(mu_t).

    By law of total variance: Var(mu) = Var(x) - P_post,
    where Var(x) = sigma_x^2/(1-phi^2) and P_post is the KF posterior variance.
    """
    K_f, _, P_post = steady_state_kalman(phi, sigma_x, sigma_o)
    var_x = sigma_x**2 / (1 - phi**2)
    return max(var_x - P_post, 1e-10)


def estimate_te(mu, e, lag=1):
    """Transfer entropy T_{mu->e} = I(mu_t; e_{t+1} | e_t) via Gaussian regression."""
    n = len(mu) - lag
    if n < 100:
        return 0.0
    e_curr, e_next, mu_curr = e[:n], e[lag : n + lag], mu[:n]

    X1 = np.column_stack([e_curr, np.ones(n)])
    r1 = e_next - X1 @ np.linalg.lstsq(X1, e_next, rcond=None)[0]
    var1 = np.var(r1)

    X2 = np.column_stack([e_curr, mu_curr, np.ones(n)])
    r2 = e_next - X2 @ np.linalg.lstsq(X2, e_next, rcond=None)[0]
    var2 = np.var(r2)

    if var2 <= 0 or var1 <= 0:
        return 0.0
    return max(0.5 * np.log(var1 / var2), 0.0)


def te_analytical(beta, K, sigma_mu_sq, sigma_e):
    """Analytical transfer entropy: TE = 0.5 * log(1 + beta^2*K^2*sigma_mu^2/sigma_e^2)."""
    return 0.5 * np.log(1 + beta**2 * K**2 * sigma_mu_sq / sigma_e**2)


def sliding_r2(x, y, window=1000, step=None):
    """Sliding-window R^2 between two time series."""
    if step is None:
        step = window // 4
    n = min(len(x), len(y))
    r2_vals, centers = [], []
    i = 0
    while True:
        start = i * step
        end = start + window
        if end > n:
            break
        r = np.corrcoef(x[start:end], y[start:end])[0, 1]
        r2_vals.append(r**2 if not np.isnan(r) else 0.0)
        centers.append((start + end) // 2)
        i += 1
    return np.array(r2_vals), np.array(centers)


# ═══════════════════════════════════════════════════════════════
# Agent A simulation (proper AIF: KF + a = K*mu)
# ═══════════════════════════════════════════════════════════════

def simulate_aif(T, phi=0.7, v=0.3, alpha=0.0, beta=1.0, K=1.0, rho=0.5,
                 sigma_x=1.0, sigma_o=1.0, sigma_e=1.0, seed=42):
    """Full AIF simulation: KF perception + linear policy action.

    Dynamics:
        x_{t+1} = phi*x_t + alpha*e_t + v + noise_x
        o_t = x_t + noise_o
        mu_{t+1} = phi*mu_t + v + K_f*(o_{t+1} - phi*mu_t - v)   [KF = VFE min]
        a_t = K * mu_t                                             [EFE-optimal in LG]
        e_{t+1} = rho*e_t + beta*a_t + noise_e
    """
    rng = np.random.default_rng(seed)
    K_f, _, _ = steady_state_kalman(phi, sigma_x, sigma_o)

    x = np.zeros(T)
    o = np.zeros(T)
    mu = np.zeros(T)
    e = np.zeros(T)

    x[0] = rng.normal(v / (1 - phi), 1.0)
    o[0] = x[0] + rng.normal(0, sigma_o)
    mu[0] = o[0]
    e[0] = rng.normal(0, 1.0)

    unstable = False
    for t in range(T - 1):
        x[t + 1] = phi * x[t] + alpha * e[t] + v + rng.normal(0, sigma_x)
        o[t + 1] = x[t + 1] + rng.normal(0, sigma_o)
        mu[t + 1] = phi * mu[t] + v + K_f * (o[t + 1] - phi * mu[t] - v)
        e[t + 1] = rho * e[t] + beta * K * mu[t] + rng.normal(0, sigma_e)
        if abs(mu[t + 1]) > 1e8 or abs(e[t + 1]) > 1e8:
            unstable = True
            break

    return {
        "x": x, "o": o, "mu": mu, "e": e,
        "K_f": K_f, "unstable": unstable,
    }


# ═══════════════════════════════════════════════════════════════
# Theory of Mind: Kalman Filter = VFE Minimization in LG
# ═══════════════════════════════════════════════════════════════

def kf_tom(z, H, phi, v, Q, R, m0, P0):
    """Agent B's Theory of Mind via active inference (VFE minimization).

    B maintains a generative model of A's belief dynamics and minimizes
    variational free energy over A's hidden state. In the linear-Gaussian
    case, this reduces to a Kalman filter -- the exact VFE-optimal solution.

    Generative model:
        s_{t+1} = phi * s_t + v + w_t,  w_t ~ N(0, Q)   [A's belief dynamics]
        z_t = H * s_t + eps_t,  eps_t ~ N(0, R)          [environment obs]
    """
    T = len(z)
    m = np.zeros(T + 1)
    P = np.zeros(T + 1)

    m[0] = m0
    P[0] = P0

    for t in range(T):
        m_pred = phi * m[t] + v
        P_pred = phi**2 * P[t] + Q
        innov = z[t] - H * m_pred
        S = H**2 * P_pred + R
        K_B = P_pred * H / S
        m[t + 1] = m_pred + K_B * innov
        P[t + 1] = (1 - K_B * H) * P_pred

    return m[1:], P[1:]


def kf_tom_steady_state(H, phi, Q, R):
    """Steady-state posterior variance for B's ToM Kalman filter.

    Solves the discrete algebraic Riccati equation (DARE).
    """
    a = H**2
    b = R * (1 - phi**2) - Q * H**2
    c_coef = -Q * R

    if a < 1e-12:
        return Q / (1 - phi**2)

    disc = b**2 - 4 * a * c_coef
    P_pred = (-b + np.sqrt(max(disc, 0))) / (2 * a)
    P_post = P_pred * R / (H**2 * P_pred + R)
    return P_post


# ═══════════════════════════════════════════════════════════════
# Core: Model-Mismatch ToM Simulation
# ═══════════════════════════════════════════════════════════════

def simulate_tom_mismatch(T, phi_A=0.7, v_A=0.3, K_A=1.0,
                          phi_B=0.7, v_B=0.3, K_B=1.0,
                          beta=2.0, rho=0.5, alpha=0.0,
                          sigma_x=1.0, sigma_o=1.0, sigma_e=1.0,
                          sigma_B=0.5, seed=42,
                          phi_A_schedule=None, K_A_schedule=None):
    """Model-mismatch ToM simulation: proper AIF Agent A + sophisticated Observer B.

    Agent A does VFE minimization (KF perception) and EFE-optimal action (a=K*mu).
    Observer B does VFE minimization (KF-based ToM) with potentially misspecified
    model parameters. The externalization horizon emerges when B's model diverges
    from A's actual dynamics.

    Args:
        phi_A, v_A, K_A: Agent A's true parameters (state persistence, drift, policy gain)
        phi_B, v_B, K_B: Observer B's assumed parameters for A
        beta, rho, alpha: Shared environment parameters (common knowledge)
        sigma_x, sigma_o, sigma_e: Noise standard deviations
        sigma_B: B's observation noise
        phi_A_schedule: Optional array of length T for time-varying phi_A
    """
    rng = np.random.default_rng(seed)

    # --- Agent A's Kalman filter (VFE minimization) ---
    phi_A_0 = phi_A_schedule[0] if phi_A_schedule is not None else phi_A
    K_f_A, P_pred_A, _ = steady_state_kalman(phi_A_0, sigma_x, sigma_o)

    # Arrays for A
    x = np.zeros(T)
    o = np.zeros(T)
    mu_A = np.zeros(T)
    e = np.zeros(T)
    a = np.zeros(T)
    e_hat = np.zeros(T)  # A's environment estimate (for alpha-aware prediction)

    # Initial conditions for A
    x[0] = rng.normal(v_A / (1 - phi_A_0), 1.0)
    o[0] = x[0] + rng.normal(0, sigma_o)
    mu_A[0] = o[0]
    e[0] = rng.normal(0, 1.0)

    # Run Agent A dynamics
    for t in range(T - 1):
        phi_t = phi_A_schedule[t] if phi_A_schedule is not None else phi_A

        # Recompute KF gain if phi changes
        if phi_A_schedule is not None:
            K_f_A, _, _ = steady_state_kalman(
                np.clip(phi_t, 0.01, 0.99), sigma_x, sigma_o)

        # Action (EFE-optimal in LG)
        K_A_t = K_A_schedule[t] if K_A_schedule is not None else K_A
        a[t] = K_A_t * mu_A[t]

        # State dynamics
        x[t + 1] = phi_t * x[t] + alpha * e[t] + v_A + rng.normal(0, sigma_x)
        o[t + 1] = x[t + 1] + rng.normal(0, sigma_o)

        # KF perception (VFE minimization)
        prediction = phi_t * mu_A[t] + v_A
        if alpha > 0:
            # A is aware of feedback: incorporates estimated environment
            prediction += alpha * e_hat[t]
        mu_A[t + 1] = prediction + K_f_A * (o[t + 1] - prediction)

        # Environment dynamics
        e[t + 1] = rho * e[t] + beta * a[t] + rng.normal(0, sigma_e)
        e_hat[t + 1] = rho * e_hat[t] + beta * a[t]

        # Instability check
        if abs(mu_A[t + 1]) > 1e8 or abs(e[t + 1]) > 1e8:
            mu_A[t + 1:] = np.nan
            e[t + 1:] = np.nan
            break

    # --- Observer B's ToM (VFE minimization with misspecified model) ---

    # B's model components derived from B's assumed parameters
    phi_B_clamped = np.clip(phi_B, 0.01, 0.99)
    K_f_B, P_pred_B, _ = steady_state_kalman(phi_B_clamped, sigma_x, sigma_o)
    Q_B = K_f_B * P_pred_B       # B's assumed innovation variance
    H_B = beta * K_B              # B's assumed observation gain
    R_B = sigma_e**2 + sigma_B**2 * (1 + rho**2)  # observation noise

    # B's prior variance
    P_prior_B = Q_B / (1 - phi_B_clamped**2) if phi_B_clamped < 1.0 else Q_B * 100

    # B observes environment with noise
    rng_B = np.random.default_rng(seed + 1000)
    o_B = e + rng_B.normal(0, sigma_B, T)

    # B extracts innovations
    n = T - 1
    z = o_B[1:] - rho * o_B[:n]

    # Run B's ToM KF (update-then-predict: z[t] carries info about mu_A[t])
    # m_B_post[t] = B's posterior estimate of mu_A[t] after observing z[t]
    m_B_prior = np.zeros(n)  # prior (before z[t])
    m_B = np.zeros(n)        # posterior (after z[t]) — this is B's best estimate
    P_B_prior = np.zeros(n)
    P_B = np.zeros(n)
    innovs = np.zeros(n)     # innovation sequence
    S_vals = np.zeros(n)     # predicted innovation variance

    m_B_prior[0] = v_B / (1 - phi_B_clamped) if phi_B_clamped < 1.0 else v_B
    P_B_prior[0] = P_prior_B

    for t in range(n):
        # UPDATE: z[t] is about mu_A[t], compare to prior estimate
        innov = z[t] - H_B * m_B_prior[t]
        S = H_B**2 * P_B_prior[t] + R_B
        K_tom = P_B_prior[t] * H_B / S
        innovs[t] = innov
        S_vals[t] = S

        m_B[t] = m_B_prior[t] + K_tom * innov
        P_B[t] = (1 - K_tom * H_B) * P_B_prior[t]

        # PREDICT: propagate posterior to next timestep
        if t < n - 1:
            m_B_prior[t + 1] = phi_B_clamped * m_B[t] + v_B
            P_B_prior[t + 1] = phi_B_clamped**2 * P_B[t] + Q_B

    # Compute metrics
    mu_short = mu_A[:n]
    te_emp = estimate_te(mu_A, e)

    # Overall R^2 (from converged tail)
    tail = min(2000, n // 2)
    r_tail = np.corrcoef(mu_short[-tail:], m_B[-tail:])[0, 1]
    r2_overall = r_tail**2 if not np.isnan(r_tail) else 0.0

    # Sliding-window R^2
    r2_win, r2_centers = sliding_r2(mu_short, m_B, window=1000)

    # Posterior uncertainty ratio
    pb_ratio = np.mean(P_B[-tail:]) / P_prior_B if P_prior_B > 0 else 1.0

    # Innovation consistency (measures model fit quality)
    innov_var_ratio = np.var(innovs[-tail:]) / np.mean(S_vals[-tail:]) if np.mean(S_vals[-tail:]) > 0 else 1.0

    return {
        "x": x, "o": o, "mu_A": mu_A, "e": e, "a": a,
        "m_B": m_B, "P_B": P_B,
        "K_f_A": K_f_A, "K_f_B": K_f_B,
        "H_B": H_B, "Q_B": Q_B, "R_B": R_B,
        "P_prior_B": P_prior_B,
        "te_empirical": te_emp,
        "r2_overall": r2_overall,
        "r2_windowed": r2_win, "r2_centers": r2_centers,
        "pb_ratio": pb_ratio,
        "innov_var_ratio": innov_var_ratio,
    }


# ═══════════════════════════════════════════════════════════════
# Figure 1: Externalization of an EFE-Optimal Agent (2 panels)
# ═══════════════════════════════════════════════════════════════

def fig1_efe_externalization():
    """TE validation + phase transition: the fundamental result."""
    print("Fig 1: Externalization of an EFE-Optimal Agent...")
    phi, v, K_pol, rho = 0.7, 0.3, 1.0, 0.5
    sigma_x, sigma_o, sigma_e = 1.0, 1.0, 1.0
    r_s = 0.5
    smq = belief_variance(phi, sigma_x, sigma_o)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Panel (a): TE vs betaK with tight empirical match (T=20000) ---
    bk_vals = np.linspace(0.05, 3.0, 60)
    te_ana = np.array([te_analytical(bk, 1.0, smq, sigma_e) for bk in bk_vals])

    bk_emp = np.linspace(0.1, 2.8, 15)
    te_emp = []
    for bk in bk_emp:
        res = simulate_aif(T=20000, phi=phi, v=v, beta=bk, K=1.0, rho=rho,
                           sigma_x=sigma_x, sigma_o=sigma_o, sigma_e=sigma_e)
        te_emp.append(estimate_te(res["mu"], res["e"]))

    axes[0].plot(bk_vals, te_ana, "k-", lw=2.5, label="Analytical: "
                 r"$\frac{1}{2}\ln(1 + \beta^2 K^2 \sigma_\mu^2 / \sigma_e^2)$")
    axes[0].scatter(bk_emp, te_emp, c="steelblue", s=45, zorder=5,
                    label=r"Empirical ($T = 20\,000$)", edgecolors="k",
                    linewidths=0.5, alpha=0.9)
    axes[0].axhline(r_s, color="gray", ls="--", lw=1.5, alpha=0.6)
    axes[0].text(0.15, r_s + 0.06, r"$r_s = 0.5$ nats", color="gray", fontsize=10)

    # Shade above/below threshold
    axes[0].fill_between(bk_vals, 0, np.minimum(te_ana, r_s),
                         color="gray", alpha=0.08)
    axes[0].fill_between(bk_vals, r_s, te_ana,
                         where=te_ana > r_s, color="steelblue", alpha=0.08)

    axes[0].set_xlabel(r"Effective coupling $\beta K$", fontsize=12)
    axes[0].set_ylabel(r"Externalization $\mathcal{E}(\mu \to e)$ (nats)", fontsize=12)
    axes[0].set_title("(a) Transfer entropy vs coupling strength", fontsize=12)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 3.0)
    axes[0].set_ylim(0, None)

    # --- Panel (b): Phase transition in SNR space ---
    snr_vals = np.logspace(-2, 2, 200)
    te_snr = 0.5 * np.log(1 + snr_vals)

    # Asymptotic regimes
    te_linear = snr_vals / 2  # SNR << 1
    te_log = 0.5 * np.log(snr_vals)  # SNR >> 1

    axes[1].plot(snr_vals, te_snr, "k-", lw=2.5, label=r"$\frac{1}{2}\ln(1 + \mathrm{SNR})$")
    axes[1].plot(snr_vals[snr_vals < 1.5], te_linear[snr_vals < 1.5],
                 "--", color="coral", lw=1.5, alpha=0.7,
                 label=r"Linear: $\mathrm{SNR}/2$")
    axes[1].plot(snr_vals[snr_vals > 0.7], te_log[snr_vals > 0.7],
                 "--", color="teal", lw=1.5, alpha=0.7,
                 label=r"Log: $\frac{1}{2}\ln(\mathrm{SNR})$")

    axes[1].axvline(1.0, color="gray", ls=":", lw=2, alpha=0.6)
    axes[1].axhline(r_s, color="gray", ls="--", lw=1.5, alpha=0.4)

    # Shade dark (left) and transparent (right) regimes
    axes[1].axvspan(1e-2, 1.0, color="indigo", alpha=0.06)
    axes[1].axvspan(1.0, 1e2, color="gold", alpha=0.06)
    axes[1].text(0.06, 1.8, "DARK", color="indigo", fontsize=13,
                 fontweight="bold", alpha=0.5)
    axes[1].text(8, 1.8, "READABLE", color="goldenrod", fontsize=13,
                 fontweight="bold", alpha=0.5)
    axes[1].text(1.15, 0.05, r"SNR $= 1$", color="gray", fontsize=9,
                 rotation=90, va="bottom")

    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"Signal-to-noise ratio $\beta^2 K^2 \sigma_\mu^2 / \sigma_e^2$",
                       fontsize=12)
    axes[1].set_ylabel(r"Externalization $\mathcal{E}$ (nats)", fontsize=12)
    axes[1].set_title("(b) Phase transition at SNR = 1", fontsize=12)
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(True, alpha=0.3, which="both")
    axes[1].set_ylim(0, 2.5)

    plt.suptitle("Externalization of an EFE-Optimal Active Inference Agent",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG_DIR, f"fig1_efe_externalization.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> Done")


# ═══════════════════════════════════════════════════════════════
# Figure 2: The Externalization Horizon (2-panel heatmap)
# ═══════════════════════════════════════════════════════════════

def _compute_cod_surface(phi_A, K_A, phi_B_vals, K_B_vals, T, beta, sigma_B,
                         seed=42):
    """Compute coefficient of determination surface: CoD = 1 - MSE/Var(mu_A).

    Unlike R^2 = corr^2 (which ignores bias and scale errors), CoD properly
    penalizes an observer whose estimates are systematically biased or
    mis-scaled — exactly what happens when K_B != K_A.
    """
    n_phi, n_K = len(phi_B_vals), len(K_B_vals)
    surface = np.zeros((n_K, n_phi))
    for i, phi_B in enumerate(phi_B_vals):
        for j, K_B in enumerate(K_B_vals):
            res = simulate_tom_mismatch(
                T=T, phi_A=phi_A, K_A=K_A,
                phi_B=phi_B, K_B=K_B,
                beta=beta, sigma_B=sigma_B, seed=seed)
            mu_A = res["mu_A"][:len(res["m_B"])]
            m_B = res["m_B"]
            tail = min(2000, len(m_B) // 2)
            mse = np.mean((mu_A[-tail:] - m_B[-tail:]) ** 2)
            var_mu = np.var(mu_A[-tail:])
            surface[j, i] = max(0.0, 1.0 - mse / var_mu)
    return surface


def fig2_horizon_heatmap():
    """The horizon as a shape in model space -- THE centerpiece visual.

    Key fix: uses coefficient of determination (1 - MSE/Var) instead of R^2
    (squared correlation). R^2 is scale-invariant, so K_B mismatch barely
    affects it — the observer's estimates are rescaled but still correlated.
    CoD properly penalizes bias and scale errors, producing a clear horizon
    contour centered on the matched model theta_A.
    """
    print("Fig 2: The Externalization Horizon (heatmaps)...")
    T = 3000
    r_s_thresh = 0.5
    n_grid = 35
    phi_A, K_A = 0.7, 1.0
    sigma_B = 0.5

    phi_B_vals = np.linspace(0.1, 0.95, n_grid)
    K_B_vals = np.linspace(0.05, 2.5, n_grid)

    # 3-panel progression: dark → horizon emerges → readable
    configs = [
        (0.7, r"(a) Weak coupling ($\beta = 0.7$)"),
        (1.5, r"(b) Moderate coupling ($\beta = 1.5$)"),
        (3.0, r"(c) Strong coupling ($\beta = 3.0$)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax_idx, (beta_val, title) in enumerate(configs):
        print(f"  Computing panel {ax_idx+1} (beta={beta_val})...")
        surface = _compute_cod_surface(
            phi_A, K_A, phi_B_vals, K_B_vals, T,
            beta=beta_val, sigma_B=sigma_B)

        ax = axes[ax_idx]
        PHI_B, K_B_GRID = np.meshgrid(phi_B_vals, K_B_vals)
        im = ax.pcolormesh(PHI_B, K_B_GRID, surface, cmap="inferno",
                           shading="auto", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.85,
                     label=r"CoD: $1 - \mathrm{MSE}/\mathrm{Var}(\mu^A)$")

        # Horizon contour (only if it exists in the surface)
        if surface.max() >= r_s_thresh:
            cs = ax.contour(PHI_B, K_B_GRID, surface, levels=[r_s_thresh],
                            colors="white", linewidths=2.5)
            if len(cs.allsegs[0]) > 0:
                ax.clabel(cs, fmt=r"$r_s$", fontsize=10, colors="white")

        # Additional contours for structure
        visible_levels = [l for l in [0.2, 0.4, 0.6, 0.8]
                          if l < surface.max()]
        if visible_levels:
            ax.contour(PHI_B, K_B_GRID, surface, levels=visible_levels,
                       colors="white", linewidths=0.5, linestyles="--",
                       alpha=0.3)

        # Mark true parameters
        ax.plot(phi_A, K_A, "+", color="white", ms=18, mew=3, zorder=10)
        ax.text(phi_A + 0.03, K_A + 0.1, r"$\boldsymbol{\theta}_A$",
                color="white", fontsize=13, fontweight="bold")

        # Region labels
        dark_frac = np.mean(surface < r_s_thresh)
        if dark_frac > 0.95:
            ax.text(0.5, 1.25, "UNREADABLE", color="white", fontsize=16,
                    fontweight="bold", alpha=0.8, ha="center",
                    transform=ax.transData)
        else:
            if dark_frac > 0.15:
                ax.text(0.2, 0.2, "DARK", color="white", fontsize=14,
                        fontweight="bold", alpha=0.8, ha="center")
            if dark_frac < 0.95:
                ax.text(phi_A, K_A + 0.5, "READABLE", color="black",
                        fontsize=11, fontweight="bold", alpha=0.7,
                        ha="center")

        ax.set_xlabel(r"Observer's assumed $\varphi_B$", fontsize=12)
        ax.set_ylabel(r"Observer's assumed $K_B$", fontsize=12)
        ax.set_title(title, fontsize=11)

        # Stats
        matched_idx = (np.argmin(np.abs(K_B_vals - K_A)),
                       np.argmin(np.abs(phi_B_vals - phi_A)))
        print(f"    dark (<{r_s_thresh}) = {dark_frac:.0%}, "
              f"peak = {surface.max():.3f}, "
              f"matched = {surface[matched_idx]:.3f}")

    plt.suptitle("The Externalization Horizon: Where VFE-Based Theory of Mind Fails",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG_DIR, f"fig2_horizon_heatmap.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> Done")


# ═══════════════════════════════════════════════════════════════
# Figure 3: Observer-Dependent ToM (3 panels)
# ═══════════════════════════════════════════════════════════════

def fig3_observer_dependent():
    """Same agent, different observers -- the key conceptual claim."""
    print("Fig 3: Observer-Dependent ToM...")
    r_s_thresh = 0.5

    # Agent A's true parameters
    phi_A, v_A, K_A = 0.7, 0.3, 1.0

    # B1: matched observer
    phi_B1, v_B1, K_B1 = 0.7, 0.3, 1.0
    # B2: strongly mismatched — high persistence + low gain
    # Observer expects a slow, weakly-coupled agent (wrong on both counts)
    phi_B2, v_B2, K_B2 = 0.95, 0.3, 0.1

    # Use beta=2.0, sigma_B=0.5 where separation is dramatic
    beta_fig = 2.0
    sigma_B_fig = 0.5
    T = 8000

    # Run both observers on same Agent A (same seed)
    res1 = simulate_tom_mismatch(T=T, phi_A=phi_A, v_A=v_A, K_A=K_A,
                                 phi_B=phi_B1, v_B=v_B1, K_B=K_B1,
                                 beta=beta_fig, sigma_B=sigma_B_fig, seed=42)
    res2 = simulate_tom_mismatch(T=T, phi_A=phi_A, v_A=v_A, K_A=K_A,
                                 phi_B=phi_B2, v_B=v_B2, K_B=K_B2,
                                 beta=beta_fig, sigma_B=sigma_B_fig, seed=42)

    print(f"  B1 (matched) R2 = {res1['r2_overall']:.3f}")
    print(f"  B2 (mismatched) R2 = {res2['r2_overall']:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # --- Panel (a): Time series (200-step window) ---
    t_start, t_end = 3000, 3200
    t_range = np.arange(t_start, min(t_end, len(res1["m_B"])))

    axes[0].plot(t_range, res1["mu_A"][t_range], "k-", lw=2, alpha=0.9,
                 label=r"$\mu^A_t$ (agent's beliefs)")
    axes[0].plot(t_range, res1["m_B"][t_range], "-", color="teal", lw=1.8,
                 alpha=0.85, label=r"$m^{B_1}_t$ (matched observer)")
    axes[0].plot(t_range, res2["m_B"][t_range], "-", color="coral", lw=1.8,
                 alpha=0.85, label=r"$m^{B_2}_t$ (mismatched observer)")
    axes[0].set_xlabel("Timestep", fontsize=12)
    axes[0].set_ylabel("Belief estimate", fontsize=12)
    axes[0].set_title("(a) Belief tracking (200-step window)", fontsize=11)
    axes[0].legend(fontsize=8, loc="best")
    axes[0].grid(True, alpha=0.3)

    # --- Panel (b): R^2 vs coupling beta for both observers ---
    beta_vals = np.linspace(0.3, 4.0, 25)
    r2_bk_1 = np.zeros(len(beta_vals))
    r2_bk_2 = np.zeros(len(beta_vals))

    for i, bk in enumerate(beta_vals):
        r1 = simulate_tom_mismatch(T=5000, phi_A=phi_A, v_A=v_A, K_A=K_A,
                                   phi_B=phi_B1, v_B=v_B1, K_B=K_B1,
                                   beta=bk, sigma_B=sigma_B_fig, seed=42)
        r2 = simulate_tom_mismatch(T=5000, phi_A=phi_A, v_A=v_A, K_A=K_A,
                                   phi_B=phi_B2, v_B=v_B2, K_B=K_B2,
                                   beta=bk, sigma_B=sigma_B_fig, seed=42)
        r2_bk_1[i] = r1["r2_overall"]
        r2_bk_2[i] = r2["r2_overall"]

    axes[1].plot(beta_vals, r2_bk_1, "o-", color="teal", ms=4, lw=2,
                 label=r"$B_1$ (matched: $K_B\!=\!1.0$, $\varphi_B\!=\!0.7$)")
    axes[1].plot(beta_vals, r2_bk_2, "s-", color="coral", ms=4, lw=2,
                 label=r"$B_2$ (mismatched: $K_B\!=\!0.1$, $\varphi_B\!=\!0.95$)")
    axes[1].axhline(r_s_thresh, color="gray", ls="--", lw=1.5, alpha=0.5)
    axes[1].text(3.5, r_s_thresh + 0.03, r"$r_s$", color="gray", fontsize=10)

    # Annotate where each crosses threshold
    for r2_arr, col, lbl in [(r2_bk_1, "teal", r"$B_1$"),
                              (r2_bk_2, "coral", r"$B_2$")]:
        crossings = np.where(np.diff(np.sign(r2_arr - r_s_thresh)))[0]
        if len(crossings) > 0:
            beta_cross = beta_vals[crossings[0]]
            axes[1].axvline(beta_cross, color=col, ls=":", alpha=0.4, lw=1)

    axes[1].set_xlabel(r"Coupling strength $\beta$", fontsize=12)
    axes[1].set_ylabel(r"$R^2$ (ToM accuracy)", fontsize=12)
    axes[1].set_title("(b) Observer-dependent horizons", fontsize=11)
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.02, 1.0)

    # --- Panel (c): Posterior uncertainty P_B / P_prior ---
    # Compute P_B/P_prior across coupling strength
    pb_ratio_1 = np.zeros(len(beta_vals))
    pb_ratio_2 = np.zeros(len(beta_vals))

    for i, bk in enumerate(beta_vals):
        r1 = simulate_tom_mismatch(T=5000, phi_A=phi_A, v_A=v_A, K_A=K_A,
                                   phi_B=phi_B1, v_B=v_B1, K_B=K_B1,
                                   beta=bk, sigma_B=sigma_B_fig, seed=42)
        r2 = simulate_tom_mismatch(T=5000, phi_A=phi_A, v_A=v_A, K_A=K_A,
                                   phi_B=phi_B2, v_B=v_B2, K_B=K_B2,
                                   beta=bk, sigma_B=sigma_B_fig, seed=42)
        pb_ratio_1[i] = r1["pb_ratio"]
        pb_ratio_2[i] = r2["pb_ratio"]

    axes[2].plot(beta_vals, pb_ratio_1, "o-", color="teal", ms=4, lw=2,
                 label=r"$B_1$: $P_B / P_{\mathrm{prior}}$")
    axes[2].plot(beta_vals, pb_ratio_2, "s-", color="coral", ms=4, lw=2,
                 label=r"$B_2$: $P_B / P_{\mathrm{prior}}$")
    axes[2].axhline(1.0, color="gray", ls="--", lw=1.5, alpha=0.5)
    axes[2].text(0.5, 1.03, "No information gain (prior)", color="gray", fontsize=8)

    axes[2].set_xlabel(r"Coupling strength $\beta$", fontsize=12)
    axes[2].set_ylabel(r"$P_B / P_{\mathrm{prior}}$", fontsize=12)
    axes[2].set_title("(c) VFE reduction: posterior vs prior uncertainty", fontsize=11)
    axes[2].legend(fontsize=8, loc="upper right")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Observer-Dependent Theory of Mind under Active Inference",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG_DIR, f"fig3_observer_dependent.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> Done")


# ═══════════════════════════════════════════════════════════════
# Figure 4: Dynamic Opacity (3 panels)
# ═══════════════════════════════════════════════════════════════

def fig4_dynamic_opacity():
    """Agent changes -> observer's model goes stale -> ToM breaks."""
    print("Fig 4: Dynamic Opacity...")
    T = 10000
    r_s_thresh = 0.5

    # A transitions: K 1.0->0.3 AND phi 0.7->0.4 via sigmoid at T/2
    sigmoid = 1 / (1 + np.exp((np.arange(T) - T / 2) / (T / 16)))
    phi_A_schedule = 0.4 + (0.7 - 0.4) * sigmoid
    K_A_schedule = 0.3 + (1.0 - 0.3) * sigmoid

    # B starts matched (phi_B=0.7, K_B=1.0), stays fixed
    res = simulate_tom_mismatch(
        T=T, phi_A=0.7, v_A=0.3, K_A=1.0,
        phi_B=0.7, v_B=0.3, K_B=1.0,
        phi_A_schedule=phi_A_schedule, K_A_schedule=K_A_schedule, seed=42)

    mu_A = res["mu_A"]
    m_B = res["m_B"]
    P_B = res["P_B"]
    n = len(m_B)

    # Sliding-window R^2
    r2_vals, r2_centers = sliding_r2(mu_A[:n], m_B, window=1000)

    # Sliding-window MSE vs reported P_B
    sw = 1000
    step = sw // 4
    mse_vals, mse_centers, pb_vals = [], [], []
    for start in range(0, n - sw, step):
        end = start + sw
        mse = np.mean((mu_A[start:end] - m_B[start:end]) ** 2)
        mse_vals.append(mse)
        pb_vals.append(np.mean(P_B[start:end]))
        mse_centers.append(start + sw // 2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # --- Panel (a): Agent A's parameters over time ---
    axes[0].plot(np.arange(T), K_A_schedule, "-", color="steelblue", lw=2.5,
                 label=r"$K_A(t)$")
    axes[0].plot(np.arange(T), phi_A_schedule, "-", color="darkorange", lw=2.5,
                 label=r"$\varphi_A(t)$")
    axes[0].axhline(1.0, color="steelblue", ls="--", lw=1, alpha=0.5)
    axes[0].axhline(0.7, color="darkorange", ls="--", lw=1, alpha=0.5)
    axes[0].text(T * 0.03, 1.03, r"$K_B = 1.0$", color="steelblue", fontsize=9)
    axes[0].text(T * 0.03, 0.73, r"$\varphi_B = 0.7$", color="darkorange", fontsize=9)
    axes[0].axvline(T / 2, color="gray", ls=":", alpha=0.5, lw=2)
    axes[0].text(T / 2 + 200, 0.55, "parameter\nchange", color="gray", fontsize=9)
    axes[0].set_xlabel("Timestep", fontsize=12)
    axes[0].set_ylabel("Parameter value", fontsize=12)
    axes[0].set_title("(a) Agent A adapts; Observer B stays fixed", fontsize=11)
    axes[0].legend(fontsize=8, loc="center right")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0.2, 1.15)

    # --- Panel (b): R^2 over time ---
    axes[1].plot(r2_centers, r2_vals, "-", color="teal", lw=2)
    axes[1].axhline(r_s_thresh, color="gray", ls="--", lw=1.5, alpha=0.5)
    axes[1].axvline(T / 2, color="gray", ls=":", alpha=0.5, lw=2)
    axes[1].fill_between(r2_centers, 0, r2_vals,
                         where=np.array(r2_vals) < r_s_thresh,
                         color="coral", alpha=0.15)
    axes[1].fill_between(r2_centers, 0, r2_vals,
                         where=np.array(r2_vals) >= r_s_thresh,
                         color="teal", alpha=0.08)
    axes[1].text(T * 0.15, 0.75, "READABLE", color="teal", fontsize=11,
                 fontweight="bold", alpha=0.6)
    axes[1].text(T * 0.7, 0.15, "DARK", color="coral", fontsize=11,
                 fontweight="bold", alpha=0.6)
    axes[1].set_xlabel("Timestep", fontsize=12)
    axes[1].set_ylabel(r"$R^2$ (ToM accuracy)", fontsize=12)
    axes[1].set_title(r"(b) $R^2$ crosses horizon after parameter change", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.02, 1.0)

    # --- Panel (c): MSE vs P_B (miscalibration gap) ---
    axes[2].plot(mse_centers, mse_vals, "-", color="coral", lw=2,
                 label=r"Actual MSE: $\langle(\mu_A - m_B)^2\rangle$")
    axes[2].plot(mse_centers, pb_vals, "-", color="teal", lw=2,
                 label=r"Reported uncertainty: $P_B$")
    axes[2].axvline(T / 2, color="gray", ls=":", alpha=0.5, lw=2)

    # Shade the miscalibration gap
    axes[2].fill_between(mse_centers, pb_vals, mse_vals,
                         where=np.array(mse_vals) > np.array(pb_vals),
                         color="coral", alpha=0.15,
                         label="Miscalibration gap")

    axes[2].set_xlabel("Timestep", fontsize=12)
    axes[2].set_ylabel("Error / Uncertainty", fontsize=12)
    axes[2].set_title("(c) Confident but wrong: VFE-based ToM miscalibration",
                      fontsize=11)
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("Dynamic Opacity: Generative Model Staleness Breaks VFE-Based ToM",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(FIG_DIR, f"fig4_dynamic_opacity.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> Done")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Externalization Horizon -- 4 Decisive Figures")
    print("=" * 60)

    # Print key parameters
    phi, sigma_x, sigma_o = 0.7, 1.0, 1.0
    K_f, _, P_post = steady_state_kalman(phi, sigma_x, sigma_o)
    smq = belief_variance(phi, sigma_x, sigma_o)
    te_base = te_analytical(1.5, 1.0, smq, 1.0)
    print(f"  K_f = {K_f:.3f}, Var(mu) = {smq:.3f}")
    print(f"  TE at beta=1.5, K=1: {te_base:.3f} nats")
    print()

    # Verify matched model at key parameter sets
    print("Verifying matched models...")
    for beta_v, sigma_B_v in [(1.5, 1.5), (3.0, 1.5), (2.0, 0.5)]:
        res_m = simulate_tom_mismatch(T=5000, beta=beta_v, sigma_B=sigma_B_v, seed=42)
        print(f"  beta={beta_v}, sigma_B={sigma_B_v}: "
              f"R2={res_m['r2_overall']:.3f}, P_B/P_prior={res_m['pb_ratio']:.3f}")
    print()

    fig1_efe_externalization()
    fig2_horizon_heatmap()
    fig3_observer_dependent()
    fig4_dynamic_opacity()

    print("\n" + "=" * 60)
    print("All 4 figures saved to:", FIG_DIR)
    print("=" * 60)
