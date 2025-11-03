# Livre_AdvancesROM_KoopmanDMD

# Supplementary Notebooks — Koopman / DMD / Mori–Zwanzig

This repository contains two Jupyter notebooks that serve as supplementary material for a chapter on Koopman theory, Extended DMD (EDMD) and Mori–Zwanzig memory closure. The notebooks demonstrate algorithms on the Stuart–Landau oscillator and include plotting and simple experiments used to produce figures.

Notebooks
- `Extended_DMD.ipynb` — construct observables, compute Koopman operator via DMD/EDMD, reconstruct data, and compare analytical eigenvalues with data-driven eigenvalues.
- `Mori-Zwanzig.ipynb` — Mori–Zwanzig with fluctuation–dissipation style memory modeling: sequence creation, kernel regression (Ridge), prediction with and without memory, and visualization of kernels.

Quick start
1. Create / activate a Python environment (example with conda):
   ```bash
   conda create -n koop_env python=3.10
   conda activate koop_env
