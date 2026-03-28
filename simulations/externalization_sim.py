"""
Externalization Horizon Simulations
====================================
Simulates the linear-Gaussian active inference agent coupled to a stochastic
environment and measures externalization (transfer entropy from beliefs to
environment), validating the closed-form analytical results.

Authors: Mahault Albarracin, Alejandro Jimenez Rodriguez
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import gaussian_kde
import os

# ─── Output directory ───────────────────────────────────────────────────────
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Core simulation
# ═══════════════════════════════════════════════════════════════════════════

def simulate_agent_environment(
    T: int = 10_000,
    v: float = 0.0,          # drift
    sigma_x: float = 1.0,    # process noise
    sigma_o: float = 1.0,    # observation noise
    K: float = 1.0,          # policy gain
    beta: float = 1.0,       # environment coupling
    rho: float = 0.5,        # environment persistence
    sigma_e: float = 1.0,    # environment noise
    seed: int = 42,
) -> dict:
    """Simulate coupled agent-environment system and return trajectories."""
    rng = np.random.default_rng(seed)

    # ── Kalman filter setup ──
    # Steady-state Kalman gain for 1D constant-velocity model
    # P_pred = P + sigma_x^2
    # K_f = P_pred / (P_pred + sigma_o^2)
    # Steady state: P = K_f * sigma_o^2
    # => P^2 + P*sigma_o^2 - sigma_x^2*sigma_o^2 - sigma_o^2*P = 0...
    # simplified: Riccati => P_ss = 0.5*(-sigma_o^2 + sqrt(sigma_o^4 + 4*sigma_x^2*sigma_o^2))
    disc = sigma_o**4 + 4 * sigma_x**2 * sigma_o**2
    P_ss = 0.5 * (-sigma_o**2 + np.sqrt(disc))
    K_f = P_ss / (P_ss + sigma_o**2)

    # Pre-allocate
    x = np.zeros(T)       # true state
    o = np.zeros(T)       # observations
    mu = np.zeros(T)      # belief (posterior mean)
    a = np.zeros(T)       # actions
    e = np.zeros(T)       # environment

    # Initial conditions
    x[0] = rng.normal(0, 1)
    o[0] = x[0] + rng.normal(0, sigma_o)
    mu[0] = o[0]  # initialize belief at first observation
    a[0] = K * mu[0]
    e[0] = rng.normal(0, 1)

    for t in range(T - 1):
        # True state evolves
        x[t + 1] = x[t] + v + rng.normal(0, sigma_x)

        # Agent observes
        o[t + 1] = x[t + 1] + rng.normal(0, sigma_o)

        # Kalman belief update
        mu_pred = mu[t] + v
        mu[t + 1] = mu_pred + K_f * (o[t + 1] - mu_pred)

        # Policy
        a[t + 1] = K * mu[t + 1]

        # Environment evolves (driven by agent actions)
        e[t + 1] = rho * e[t] + beta * a[t] + rng.normal(0, sigma_e)

    return {
        "x": x, "o": o, "mu": mu, "a": a, "e": e,
        "K_f": K_f, "P_ss": P_ss,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Externalization estimation
# ═══════════════════════════════════════════════════════════════════════════

def estimate_transfer_entropy_gaussian(mu, e, lag=1):
    """
    Estimate transfer entropy T_{mu -> e} = I(mu_t; e_{t+1} | e_t)
    using the Gaussian assumption (exact for linear-Gaussian systems).

    T = 0.5 * log( Var(e_{t+1} | e_t) / Var(e_{t+1} | e_t, mu_t) )
    """
    n = len(mu) - lag

    e_curr = e[:n]
    e_next = e[lag:n + lag]
    mu_curr = mu[:n]

    # Var(e_{t+1} | e_t) via linear regression residuals
    X1 = np.column_stack([e_curr, np.ones(n)])
    beta1, _, _, _ = np.linalg.lstsq(X1, e_next, rcond=None)
    resid1 = e_next - X1 @ beta1
    var_cond_e = np.var(resid1)

    # Var(e_{t+1} | e_t, mu_t) via linear regression residuals
    X2 = np.column_stack([e_curr, mu_curr, np.ones(n)])
    beta2, _, _, _ = np.linalg.lstsq(X2, e_next, rcond=None)
    resid2 = e_next - X2 @ beta2
    var_cond_e_mu = np.var(resid2)

    # Transfer entropy
    if var_cond_e_mu <= 0 or var_cond_e <= 0:
        return 0.0
    te = 0.5 * np.log(var_cond_e / var_cond_e_mu)
    return max(te, 0.0)


def conditional_var_mu_given_e(mu, e):
    """Estimate Var(mu_t | e_t) via linear regression residuals."""
    n = len(mu)
    X = np.column_stack([e, np.ones(n)])
    beta_hat, _, _, _ = np.linalg.lstsq(X, mu, rcond=None)
    resid = mu - X @ beta_hat
    return np.var(resid)


def analytical_externalization(beta, K, sigma_mu, sigma_e):
    """Closed-form externalization: E = 0.5 * log(1 + (beta*K)^2 * sigma_mu^2 / sigma_e^2)"""
    snr = (beta * K) ** 2 * sigma_mu**2 / sigma_e**2
    return 0.5 * np.log(1 + snr)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Phase Transition
# ═══════════════════════════════════════════════════════════════════════════

def fig1_phase_transition():
    """E vs SNR showing the phase transition at SNR ≈ 1."""
    print("Generating Figure 1: Phase Transition...")

    snr_values = np.logspace(-2, 2, 50)
    empirical = []
    analytical = []

    sigma_e_base = 1.0
    sigma_x = 1.0
    sigma_o = 1.0

    for snr in snr_values:
        # Choose beta*K to achieve desired SNR (with sigma_mu_cond ≈ 1)
        bk = np.sqrt(snr * sigma_e_base**2)
        K_val = 1.0
        beta_val = bk / K_val if K_val > 0 else 0.0

        result = simulate_agent_environment(
            T=50_000, K=K_val, beta=beta_val,
            sigma_e=sigma_e_base, sigma_x=sigma_x, sigma_o=sigma_o,
            seed=42,
        )

        te = estimate_transfer_entropy_gaussian(result["mu"], result["e"])
        empirical.append(te)

        # Use CONDITIONAL variance Var(mu_t | e_t)
        sigma_mu_cond = np.sqrt(conditional_var_mu_given_e(result["mu"], result["e"]))
        analytical.append(analytical_externalization(beta_val, K_val, sigma_mu_cond, sigma_e_base))

    # Theoretical curve
    snr_smooth = np.logspace(-2, 2, 500)
    theory = 0.5 * np.log(1 + snr_smooth)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(snr_smooth, theory, "k-", linewidth=2, label=r"$\frac{1}{2}\log(1 + \mathrm{SNR})$")
    ax.scatter(snr_values, empirical, c="steelblue", s=30, zorder=5, label="Empirical (Gaussian TE)")
    ax.scatter(snr_values, analytical, c="orange", s=20, marker="x", zorder=5,
              label="Analytical (conditional $\\sigma_\\mu$)")
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label=r"$\mathrm{SNR} = 1$ (critical)")
    ax.axhspan(0, 0.1, color="gray", alpha=0.15, label="Dark regime")

    ax.set_xscale("log")
    ax.set_xlabel(r"Signal-to-Noise Ratio: $\beta^2 K^2 \sigma_\mu^2 / \sigma_e^2$", fontsize=12)
    ax.set_ylabel(r"Externalization $\mathcal{E}$ (nats)", fontsize=12)
    ax.set_title("Phase Transition in Externalization", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "phase_transition.pdf"), dpi=150)
    plt.savefig(os.path.join(FIG_DIR, "phase_transition.png"), dpi=150)
    plt.close()
    print("  -> Saved phase_transition.pdf/png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Horizon Surface
# ═══════════════════════════════════════════════════════════════════════════

def fig2_horizon_surface():
    """Externalization as a function of beta*K and sigma_e."""
    print("Generating Figure 2: Horizon Surface...")

    bk_range = np.linspace(0.01, 5.0, 60)
    se_range = np.linspace(0.1, 5.0, 60)
    BK, SE = np.meshgrid(bk_range, se_range)

    # Analytical externalization (assume sigma_mu ≈ 1)
    sigma_mu = 1.0
    E_surface = 0.5 * np.log(1 + (BK**2 * sigma_mu**2) / SE**2)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    cmap = cm.viridis
    im = ax.pcolormesh(BK, SE, E_surface, cmap=cmap, shading="auto")
    plt.colorbar(im, ax=ax, label=r"Externalization $\mathcal{E}$ (nats)")

    # Horizon contour at r_s = 0.5 nats
    r_s = 0.5
    cs = ax.contour(BK, SE, E_surface, levels=[r_s], colors="white", linewidths=2)
    ax.clabel(cs, fmt=r"$r_s = %.1f$" % r_s, fontsize=10, colors="white")

    # Additional contours
    cs2 = ax.contour(BK, SE, E_surface, levels=[0.1, 0.25, 1.0, 2.0],
                     colors="white", linewidths=0.5, linestyles="--", alpha=0.5)

    ax.set_xlabel(r"Effective coupling $\beta K$", fontsize=12)
    ax.set_ylabel(r"Environment noise $\sigma_e$", fontsize=12)
    ax.set_title("Externalization Horizon Surface", fontsize=14)

    # Annotate regions
    ax.text(0.5, 4.0, "DARK\n(opaque)", color="white", fontsize=14, ha="center",
            fontweight="bold", alpha=0.8)
    ax.text(4.0, 0.5, "TRANSPARENT\n(readable)", color="black", fontsize=14, ha="center",
            fontweight="bold", alpha=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "horizon_surface.pdf"), dpi=150)
    plt.savefig(os.path.join(FIG_DIR, "horizon_surface.png"), dpi=150)
    plt.close()
    print("  -> Saved horizon_surface.pdf/png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Multi-Agent — Screen vs Direct Access
# ═══════════════════════════════════════════════════════════════════════════

def fig3_multi_agent():
    """Compare screen (environment-mediated) vs direct behavioral access."""
    print("Generating Figure 3: Multi-Agent Scenarios...")

    sigma_e_range = np.linspace(0.1, 5.0, 50)
    K_val = 1.0
    beta_val = 2.0
    sigma_mu = 1.0
    sigma_B = 0.5    # observer noise (screen scenario)
    sigma_eta = 0.3   # action noise (direct scenario)

    e_screen = []
    e_direct = []
    e_total = []

    for se in sigma_e_range:
        # Total externalization to environment
        e_tot = 0.5 * np.log(1 + (beta_val * K_val)**2 * sigma_mu**2 / se**2)
        e_total.append(e_tot)

        # Screen: observer sees environment + own noise
        e_scr = 0.5 * np.log(1 + (beta_val * K_val)**2 * sigma_mu**2 / (se**2 + sigma_B**2))
        e_screen.append(e_scr)

        # Direct: observer sees actions + action noise (independent of sigma_e)
        e_dir = 0.5 * np.log(1 + K_val**2 * sigma_mu**2 / sigma_eta**2)
        e_direct.append(e_dir)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(sigma_e_range, e_total, "k-", linewidth=2, label=r"Total $\mathcal{E}(\mu \to e)$")
    ax.plot(sigma_e_range, e_screen, "b--", linewidth=2,
            label=r"Screen: $\mathcal{E}_{\mathrm{screen}}$ ($\sigma_B=%.1f$)" % sigma_B)
    ax.plot(sigma_e_range, e_direct, "r:", linewidth=2,
            label=r"Direct: $\mathcal{E}_{\mathrm{direct}}$ ($\sigma_\eta=%.1f$)" % sigma_eta)

    # Threshold
    r_s = 0.5
    ax.axhline(y=r_s, color="gray", linestyle="-.", alpha=0.5, label=r"Threshold $r_s = %.1f$" % r_s)
    ax.axhspan(0, r_s, color="gray", alpha=0.1)

    ax.set_xlabel(r"Environment noise $\sigma_e$", fontsize=12)
    ax.set_ylabel(r"Observable externalization (nats)", fontsize=12)
    ax.set_title(r"Multi-Agent: Screen vs Direct Access ($\beta=%.1f, K=%.1f$)" % (beta_val, K_val),
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "multi_agent.pdf"), dpi=150)
    plt.savefig(os.path.join(FIG_DIR, "multi_agent.png"), dpi=150)
    plt.close()
    print("  -> Saved multi_agent.pdf/png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Empirical Validation — Sweep over beta*K
# ═══════════════════════════════════════════════════════════════════════════

def fig4_coupling_sweep():
    """Sweep beta*K and compare empirical TE to analytical prediction."""
    print("Generating Figure 4: Coupling Sweep Validation...")

    bk_values = np.linspace(0.0, 4.0, 30)
    sigma_e = 1.0
    T = 30_000

    empirical_te = []
    analytical_e = []

    for bk in bk_values:
        K_val = 1.0
        beta_val = bk / K_val if K_val > 0 else 0.0

        result = simulate_agent_environment(
            T=T, K=K_val, beta=beta_val, sigma_e=sigma_e, seed=42
        )

        te = estimate_transfer_entropy_gaussian(result["mu"], result["e"])
        empirical_te.append(te)

        # Use CONDITIONAL variance Var(mu_t | e_t)
        sigma_mu_cond = np.sqrt(conditional_var_mu_given_e(result["mu"], result["e"]))
        ae = analytical_externalization(beta_val, K_val, sigma_mu_cond, sigma_e)
        analytical_e.append(ae)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(bk_values, analytical_e, "k-", linewidth=2, label="Analytical")
    ax.scatter(bk_values, empirical_te, c="steelblue", s=40, zorder=5, label="Empirical (Gaussian TE)")

    # Mark the dark region
    ax.axhspan(0, 0.1, color="gray", alpha=0.15)
    ax.text(0.3, 0.05, "Dark", fontsize=10, color="gray", va="center")

    ax.set_xlabel(r"Effective coupling $\beta K$", fontsize=12)
    ax.set_ylabel(r"Externalization $\mathcal{E}$ (nats)", fontsize=12)
    ax.set_title("Externalization vs Coupling Strength", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "coupling_sweep.pdf"), dpi=150)
    plt.savefig(os.path.join(FIG_DIR, "coupling_sweep.png"), dpi=150)
    plt.close()
    print("  -> Saved coupling_sweep.pdf/png")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5: Trajectory visualization
# ═══════════════════════════════════════════════════════════════════════════

def fig5_trajectories():
    """Show agent belief and environment trajectories for dark vs transparent regimes."""
    print("Generating Figure 5: Trajectory Comparison...")

    T = 500

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    configs = [
        {"beta": 0.1, "K": 0.1, "sigma_e": 2.0, "label": "Dark regime ($\\beta K = 0.01$, $\\sigma_e = 2.0$)"},
        {"beta": 3.0, "K": 1.5, "sigma_e": 0.5, "label": "Transparent regime ($\\beta K = 4.5$, $\\sigma_e = 0.5$)"},
    ]

    for ax, cfg in zip(axes, configs):
        result = simulate_agent_environment(
            T=T, K=cfg["K"], beta=cfg["beta"], sigma_e=cfg["sigma_e"], seed=42
        )

        t_axis = np.arange(T)
        ax.plot(t_axis, result["mu"], "b-", alpha=0.8, linewidth=1, label=r"Belief $\mu_t$")
        ax.plot(t_axis, result["e"], "r-", alpha=0.6, linewidth=1, label=r"Environment $e_t$")

        te = estimate_transfer_entropy_gaussian(result["mu"], result["e"])
        ax.set_title(f'{cfg["label"]}  |  $\\mathcal{{E}} = {te:.3f}$ nats', fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_ylabel("State value")

    axes[-1].set_xlabel("Timestep", fontsize=12)
    plt.suptitle("Agent Belief vs Environment: Dark and Transparent Regimes", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "trajectories.pdf"), dpi=150)
    plt.savefig(os.path.join(FIG_DIR, "trajectories.png"), dpi=150)
    plt.close()
    print("  -> Saved trajectories.pdf/png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Externalization Horizon — Simulation Suite")
    print("=" * 60)

    fig1_phase_transition()
    fig2_horizon_surface()
    fig3_multi_agent()
    fig4_coupling_sweep()
    fig5_trajectories()

    print("\n" + "=" * 60)
    print("All figures saved to:", FIG_DIR)
    print("=" * 60)
