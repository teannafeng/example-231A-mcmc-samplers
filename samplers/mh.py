"""
Metropolis-Hastings (random-walk) sampler.

    x' = x + sigma * eta,  eta ~ N(0, I)

Each iteration:
    1. Accept proposal with probability min(1, exp(-(U(x') - U(x)))).
"""
import numpy as np


def step(x, U, sigma, rng):
    """
    Run one MH step. 

    Returns
    -------
    A tuple (new_x, accepted, proposal).
    """
    x_new = x + sigma * rng.standard_normal(x.shape)
    log_alpha = -(U(x_new) - U(x))
    accepted = np.log(rng.random()) < log_alpha
    return (
        x_new if accepted else x, 
        accepted, 
        x_new
    )

SAMPLER_MH = {
    'key': 'mh',
    'label': 'MH',
    'sliders': [
        {'id': 'sigma', 'label': 'proposal σ', 'min': 0.01, 'max': 1.0, 'step': 0.01, 'default': 0.50},
    ],
    'right_panel': 'mh',
    'iterate_js': r"""
    const sigma = cfg.sigma;
    const [eta0, eta1] = gaussianPair(rng);
    const px0 = state.x0 + sigma * eta0;
    const px1 = state.x1 + sigma * eta1;
    const logAlpha = -(Ufn(px0, px1) - Ufn(state.x0, state.x1));
    const accept = Math.log(rng()) < logAlpha;
    return {
      frames: [[state.x0, state.x1], [px0, px1]],
      end: [px0, px1],
      accept,
      nextState: accept ? { x0: px0, x1: px1 } : state,
      mh: {
        start: [state.x0, state.x1],
        proposal: [px0, px1],
        sigma: sigma,
      },
    };
"""
}


if __name__ == '__main__':
    pass