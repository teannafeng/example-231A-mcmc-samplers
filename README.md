# EDUC 231A - Introduction to Gradient-Based MCMC Samplers

This repository accompanies a guest lecture given for EDUC 231A — Toolkit for Quantitative Methods Research on 2026-05-21 at UCLA.

## Materials

- Slides: `./slides`
- Key references: `./references/`
- Sampler code shown in slides: `./samplers`

## Interactive demo

Set up a virtual environment and install package dependencies:

```
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install numpy matplotlib
```

Generate the demo (run from the repo root):

```
python webgl_demo.py
```

This writes `./demo/index.html`. Open it in a browser to interact with the WebGL demo. 

## Folder structure

```
example-231A-mcmc-samplers/
│
├── samplers/                          
│   ├── mh.py                  # Metropolis-Hastings (MH)
│   ├── mala.py                # Metropolis-adjusted Langevin Algorithm (MALA)
│   └── hmc.py                 # Hamiltonian Monte Carlo (HMC)
│
├── slides/                    # lecture slides
├── references/                # key literature
├── demo/                      # output: index.html (generated)
│
├── targets.py                 # target distribution
├── webgl_scene.py             # HTML/JS template + build_html()
├── webgl_demo.py              # entry point to build demo/index.html
│
├── .gitattributes
├── .gitignore
└── README.md
```


