"""
Generate the demo HTML file.

Run with:
    python webgl_demo.py

Save the file to:
    ./demo/index.html
"""
import os
from targets import TARGETS
from samplers.mh import SAMPLER_MH
from samplers.mala import SAMPLER_MALA
from samplers.hmc import SAMPLER_HMC
from webgl_scene import build_html


if __name__ == '__main__':
    out_dir = './demo/'
    os.makedirs(out_dir, exist_ok=True)

    target = TARGETS['2d_normal']

    html = build_html(
        target,
        samplers=[SAMPLER_MH, SAMPLER_MALA, SAMPLER_HMC],
        default_sampler='mh',
        default_rho=0.50,
        n_iter=200,
        default_speed_ms=80,
    )
    out_path = os.path.join(out_dir, 'index.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Saved {out_path} (open in a browser to view)')