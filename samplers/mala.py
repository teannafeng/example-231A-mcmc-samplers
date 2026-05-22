"""
Metropolis-adjusted Langevin algorithm (MALA):

    x' = x + (eps^2 / 2) * grad_U(x) + eps * z,  z ~ N(0, I)

Each itereation:
    1. Accept with alpha = min(1, exp(-(U(x') - U(x)) + log_q(x | x') - log_q(x' | x))).
"""
import numpy as np


def step(x, U, grad_U, eps, rng):
    """
    Run one MALA step. 
    
    Returns
    -------
    A tuple (new_x, accepted, proposal).
    """
    g = grad_U(x)
    mu_fwd = x - 0.5 * eps**2 * g
    x_new = mu_fwd + eps * rng.standard_normal(x.shape)

    g_new = grad_U(x_new)
    mu_bwd = x_new - 0.5 * eps**2 * g_new

    log_q_fwd = -0.5 * np.sum((x_new - mu_fwd)**2) / eps**2
    log_q_bwd = -0.5 * np.sum((x - mu_bwd)**2) / eps**2

    log_alpha = -(U(x_new) - U(x)) + (log_q_bwd - log_q_fwd)
    accepted = np.log(rng.random()) < log_alpha
    return (x_new if accepted else x, accepted, x_new)

SAMPLER_MALA = {
    'key': 'mala',
    'label': 'Langevin (MALA)',
    'sliders': [
        {'id': 'eps', 'label': 'step size ε', 'min': 0.01, 'max': 1.00, 'step': 0.01, 'default': 0.50},
    ],
    'right_panel': 'mala',
    'iterate_js': r"""
    const eps = cfg.eps;
    const eps2 = eps * eps;
    const gFwd = gradUfn(state.x0, state.x1);
    const muFwd0 = state.x0 - 0.5 * eps2 * gFwd[0];
    const muFwd1 = state.x1 - 0.5 * eps2 * gFwd[1];
    const [eta0, eta1] = gaussianPair(rng);
    const px0 = muFwd0 + eps * eta0;
    const px1 = muFwd1 + eps * eta1;

    const gRev = gradUfn(px0, px1);
    const muRev0 = px0 - 0.5 * eps2 * gRev[0];
    const muRev1 = px1 - 0.5 * eps2 * gRev[1];

    const dFwd0 = px0 - muFwd0, dFwd1 = px1 - muFwd1;
    const dRev0 = state.x0 - muRev0, dRev1 = state.x1 - muRev1;
    const logQfwd = -0.5 * (dFwd0*dFwd0 + dFwd1*dFwd1) / eps2;
    const logQrev = -0.5 * (dRev0*dRev0 + dRev1*dRev1) / eps2;

    const logAlpha = -(Ufn(px0, px1) - Ufn(state.x0, state.x1)) + (logQrev - logQfwd);
    const accept = Math.log(rng()) < logAlpha;
    return {
      frames: [[state.x0, state.x1], [px0, px1]],
      end: [px0, px1],
      accept,
      nextState: accept ? { x0: px0, x1: px1 } : state,
      mala: {
        start: [state.x0, state.x1],
        muFwd: [muFwd0, muFwd1],
        muRev: [muRev0, muRev1],
        proposal: [px0, px1],
        eps: eps,
      },
    };
"""
}

if __name__ == '__main__':
    pass