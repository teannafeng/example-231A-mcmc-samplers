"""
Hamiltonian Monte Carlo (HMC):

    H(x, m) = U(x) + 0.5 * ||m||^2

Each iteration:
    1. Resample m ~ N(0, I).
    2. Simulate L leapfrog steps.
    3. Accept the endpoint (x', m') with probability min(1, exp(-(H(x', m') - H(x, m)))).
"""

import numpy as np

def leapfrog(x, m, grad_U, eps, L):
    x, m = x.copy(), m.copy()
    traj = [x.copy()]
    for l in range(L):
        m -= 0.5 * eps * grad_U(x)   # half step in m
        x += eps * m                 # full step in x
        m -= 0.5 * eps * grad_U(x)   # half step in m
        traj.append(x.copy())
    return x, m, traj

def step(x, U, grad_U, eps, L, rng):
    """
    Run one HMC step. 
    
    Returns
    -------
    A tuple (new_x, accepted, proposal, trajectory).
    """
    m = rng.standard_normal(x.shape)
    H0 = U(x) + 0.5 * np.sum(m**2)
    x_new, m_new, traj = leapfrog(x, m, grad_U, eps, L)
    H1 = U(x_new) + 0.5 * np.sum(m_new**2)
    log_alpha = -(H1 - H0)
    accepted = np.log(rng.random()) < log_alpha
    return (x_new if accepted else x, accepted, x_new, traj)

SAMPLER_HMC = {
    'key': 'hmc',
    'label': 'HMC',
    'sliders': [
        {'id': 'L',   'label': 'trajectory L',  'min': 1,    'max': 50,   'step': 1,    'default': 20},
        {'id': 'eps', 'label': 'step size ε',   'min': 0.01, 'max': 1.00, 'step': 0.01, 'default': 0.5},
    ],
    'right_panel': 'momentum',
    'iterate_js': r"""
    const eps = cfg.eps, L = cfg.L;
    const [m0i, m1i] = gaussianPair(rng);
    let m0 = m0i, m1 = m1i;
    const E0 = Ufn(state.x0, state.x1) + 0.5 * (m0*m0 + m1*m1);

    let x0 = state.x0, x1 = state.x1;
    const traj = [[x0, x1]];
    const momenta = [[m0, m1]];

    for (let i = 0; i < L; i++) {
      let g = gradUfn(x0, x1);
      m0 -= 0.5 * eps * g[0];
      m1 -= 0.5 * eps * g[1];
      traj.push([x0, x1]);
      momenta.push([m0, m1]);

      x0 += eps * m0;
      x1 += eps * m1;
      traj.push([x0, x1]);
      momenta.push([m0, m1]);

      g = gradUfn(x0, x1);
      m0 -= 0.5 * eps * g[0];
      m1 -= 0.5 * eps * g[1];
      traj.push([x0, x1]);
      momenta.push([m0, m1]);
    }

    const E1 = Ufn(x0, x1) + 0.5 * (m0*m0 + m1*m1);
    const logAlpha = -(E1 - E0);
    const accept = Math.log(rng()) < logAlpha;
    return {
      frames: traj,
      momenta: momenta,
      end: [x0, x1],
      accept,
      nextState: accept ? { x0, x1 } : state,
    };
"""
}


if __name__ == '__main__':
    pass