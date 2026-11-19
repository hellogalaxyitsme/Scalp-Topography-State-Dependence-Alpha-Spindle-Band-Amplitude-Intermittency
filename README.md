# Scalp topography and state dependence of alpha and spindle-band amplitude intermittency

Analysis code for estimating the amplitude dynamics of narrow-band EEG rhythms with a stochastic normal-form model.

The complex amplitude of a rhythm is modeled by the stochastic Bautin (generalized Hopf) normal form. Its three scale-free coordinates are:
- the distance to the Hopf bifurcation, μ̃;
- the criticality, c̃;
- the amplitude timescale, t₀.

These coordinates are estimated at every scalp channel by simulation-based inference. The simulations pass through the same filtering, background and record length as the data. Every estimate is compared with IAAFT surrogates ("linear twins") of the same recording.

The repository covers every step from raw recordings to numerical results:
- simulation banks and amortized posteriors;
- empirical summary statistics;
- scalp topography with montage and peak-selection variants;
- burst predictive checks;
- comparison with direct envelope statistics;
- displacements along a common state axis;
- discrimination between intrinsic and modulated amplitude dynamics.

## Repository structure

```
ampint/                 Python package: model, inference statistics, signal processing, loaders, statistics
scripts/                analysis steps (one script per step, run from the repository root)
config/                 fixed inputs (list of simulation-bank chunks used to train the alpha posterior)
run_pipeline.sh         complete sequence of analysis steps, in order
pyproject.toml          package definition
requirements.txt        exact versions of the reference environment
```

### Package `ampint`

| Module | Content |
|---|---|
| `nf.py` | Stochastic Bautin normal form: stationary power density, closed-form estimates with and without background deconvolution, fold distance, regimes, Numba Euler–Maruyama integrators |
| `fp.py` | One-dimensional power diffusion: generator, power autocorrelation, amplitude timescale |
| `spec.py` | Spectrum, aperiodic fit, peak frequency, band-pass analytic signal, background fraction, per-run IAAFT surrogates |
| `summ.py` | The 91 summary statistics used for inference (`FEATURES`) |
| `pipeline.py` | Channel-level spectrum on artifact-free data |
| `bursts.py` | Burst detection and burst statistics |
| `dfa.py` | Detrended fluctuation analysis of the amplitude envelope within artifact-free runs |
| `stats.py` | Normative models, bootstrap and permutation statistics, false discovery rate, channel adjacency, cluster-based permutation tests (paired and covariate-adjusted with Freedman–Lane sign flipping) |
| `csd.py` | Surface Laplacian as a linear operator (spherical splines, MNE) |
| `chaos.py` | Surrogate-referenced complexity indices, computed together with the spectral covariates |
| `io_tdbrain.py`, `io_sleepedf.py`, `io_anphy.py` | Dataset loaders with artifact masks (TDBRAIN also with ocular-regression and surface-Laplacian variants) |
| `paths.py` | Dataset locations |

## Installation

Python 3.10 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Data

All datasets are public. Set the environment variables below, or place the data at the default locations under `./data` defined in `ampint/paths.py`.

| Dataset | Source | Environment variable(s) |
|---|---|---|
| TDBRAIN, release V3.1 (resting EEG and participants table) | Brainclinics Foundation (registration required) | `TDBRAIN_ROOT`, `TDBRAIN_META` |
| Sleep-EDF Expanded, sleep-cassette study | PhysioNet | `SLEEPEDF_ROOT` |
| ANPHY-Sleep | Open Science Framework, doi:10.17605/OSF.IO/R26FH | `ANPHY_ROOT`; working directories `ANPHY_TMP`, `ANPHY_CACHE` |
| MDD patients and healthy controls EEG | figshare, doi:10.6084/m9.figshare.4244171 | `MUMTAZ_ROOT` |
| EEG: Depression rest | OpenNeuro ds003478, version 1.1.0 | `DS003478_ROOT` |

ANPHY-Sleep nights are converted once to a 250-Hz average-reference cache (about 60 GB for 28 nights). The temporary extract of each night is deleted after conversion, and the original files are only read.

## Running the analyses

All scripts are run from the repository root. They read and write `results/`, which is created on first use. `run_pipeline.sh` runs every step in order:

```bash
NJ=16 bash run_pipeline.sh      # NJ: number of worker processes for the parallel steps
```

| Stage | Scripts | Main outputs |
|---|---|---|
| 1. Simulation banks and amortized posteriors | `sbi_bank.py`, `sbi_fit.py`, `eval_mod.py`, `sbi_diagnostics.py`, `delta_relations.py` | `sbi_<config>_full.joblib`, held-out predictions, validation reports |
| 2. Empirical summary statistics | `run_nf_tdbrain.py`, `run_nf_tdbrain_variants.py`, `run_chaos_tdbrain.py`, `run_nf_sleep.py`, `run_nf_anphy.py`, `run_mumtaz.py`, `run_ds003478.py` | `nf_*`, spectral covariates |
| 3. Posteriors and simulated support | `nf_apply.py`, `ood_matched.py`, `coverage_rows.py`, `coverage_anphy.py` | `post_*.csv`, `coverage_*.csv` |
| 4. Dataset analyses | `analyze_nf_tdbrain.py`, `analyze_sleep.py`, `analyze_anphy.py`, `analyze_mumtaz.py`, `analyze_ds003478.py`, `fold_stats.py`, `fold_calibration.py` | per-dataset summaries, fold calibration |
| 5. Topography, montage variants and peak selection | `topo_tdbrain.py`, `snr_adjusted_topo.py`, `variants_summary.py`, `peak_sensitivity.py`, `twin_checks.py` | `topo_tdbrain*.json`, `snradj_tdbrain.json`, `peak_sensitivity.json`, `twin_checks.json` |
| 6. Bursts, direct envelope statistics and direct-fit baseline | `run_bursts_tdbrain.py`, `ppc_bursts.py`, `ppc_stability.py`, `analyze_bursts.py`, `analyze_bursts_retest.py`, `descriptor_benchmark.py`, `direct_fit_baseline.py` | `bursts_*`, `ppc_stability.json`, `descriptor_benchmark.json`, `direct_fit_baseline.json` |
| 7. State axis | `displacements.py`, `state_axis.py`, `axis_snr_check.py`, `sim_alternatives.py`, `axis_perturbations.py`, `robustness_axis.py`, `state_axis_joint.py`, `axis_covariates.py` | `state_axis_joint.json`, `robustness_axis_joint.json` |
| 8. Mechanism discrimination | `mech_classifier.py`, `mech_domain.py`, `sbi_twinbank.py`, `mech_twin.py`, `mech_sleep_control.py`, `mech_twin_broad.py`, `mech_twin_summary.py`, `mech_state_checks.py` | `mech_twin_broad.json`, `mechtwb_*.csv`, `mech_state_checks.json` |
| 9. Analysis populations | `population_counts.py` | `population_counts.json` |
| 10. Sensitivity analyses | `tdbrain_checks.py`, `sleep_checks.py`, `delta_direct.py`, `ppc_recovery.py`, `ppc_posterior.py`, `sensitivity_checks.py`, `spatial_benchmark.py`, `adjusted_topo_spectral.py`, `mech_nearfold.py`, `metadata_checks.py` | `post_tdbrain_checks.csv`, `post_sleep_checks_*.csv`, `delta_direct.json`, `bursts_ppc_recovery.csv`, `bursts_ppc_post*.csv`, `sensitivity_checks.json`, `spatial_benchmark.json`, `adjusted_topo_spectral.json`, `mech_nearfold.json`, `metadata_checks.json` |
| 11. Fold bias from parameter fluctuations | `fold_bias_theory.py`, `fold_shift_direction.py` | `fold_bias_theory.json`, `fold_bias_theory_sim.csv`, `fold_shift_direction.json`, `fold_shift_direction_map.csv`, `fold_shift_direction_sim.csv` |
| 12. Integrator accuracy, initialization and observation process | `integrator_check.py`, `init_check.py`, `obs_fragmentation.py`, `obs_fixedband.py` | `integrator_check.json`, `obs_frag_masks.json`, `obs_fragmentation.csv`, `obs_fragmentation.json`, `obs_fixedband.csv`, `obs_fixedband.json`, `init_check_spot.json`, `init_check.csv`, `init_check_paired.csv`, `init_check.json` |

Notes on stage 8:
- `mech_classifier.py` and `mech_domain.py` train a classifier on the raw summary statistics. They also show that simulations and empirical surrogates are separable on these statistics.
- The empirical classifiers are therefore trained on differences between each signal and its own IAAFT twin.
- `mech_twin_broad.py` contrasts two intrinsic classes, additive noise and state-dependent noise, with fast modulation of excitability. It also trains a three-class version and evaluates empirical negative controls (the difference between two independent twins).
- `mech_state_checks.py` tests whether the state changes of the empirical probabilities persist at matched fold distance, for credible spectral peaks, within the simulated support and at zero change in the estimated signal-to-background ratio.
- `mech_drivers.py` groups the summary statistics into families and measures how much each family contributes to the classifier and to the separation of EEG from simulations.

Stage 10:
- `tdbrain_checks.py` builds linear twins whose peak frequency and background fraction are re-estimated on the twin, and computes the DFA exponent of the alpha envelope.
- `sleep_checks.py` recomputes the spindle-band statistics in a fixed 11.5–14.5 Hz band, independent of the selected peak.
- `delta_direct.py` trains a direct estimator of the fold distance and compares it with the plug-in estimate from the posterior medians.
- `ppc_recovery.py` and `ppc_posterior.py` test the burst predictions in simulations with known parameters and with draws from the posterior.
- `sensitivity_checks.py` collects these analyses with cluster tests on standardized effects, state-axis variants, sleep derivations and the μ̃ effects.
- `spatial_benchmark.py` compares the scalp maps of the state effects for the model coordinates and the direct statistics.
- `adjusted_topo_spectral.py` repeats the SNR-adjusted eye-opening map with directly measured spectral covariates and reports the recovery of the estimated SNR.
- `mech_nearfold.py` selects near-fold EEG and simulations with one interval and cross-validates the domain classifiers with participant-grouped folds.
- `metadata_checks.py` describes the test-retest interval and treatment records and repeats the MDD contrast for participants with a recorded formal diagnosis.

Stage 11:
- `fold_bias_theory.py` computes the constant-parameter coordinates closest (by moment matching) to the stationary power density averaged over Gaussian fluctuations of the bifurcation parameter, compares them with the second-order closed form, and simulates the normal form with Ornstein-Uhlenbeck modulation of the bifurcation parameter over a range of correlation times.
- `fold_shift_direction.py` maps the exact shift of the fold coordinate over the noisy focus for two fluctuation strengths, compares its sign with the leading-order expression, and repeats the modulated simulations for a shorter record and correlation times up to the record length.

Stage 12:
- `integrator_check.py` integrates the normal form with the step rule of the simulation banks, and with steps 4 and 16 times smaller, and compares the power distribution with the exact stationary density.
- `init_check.py` simulates the same parameters and noise from the bank initialization and from the exact stationary density, passes both through the pipeline, the alpha posterior and the main classifier, and compares the estimates by regime.
- `obs_fragmentation.py` cuts simulated signals with the artifact masks of TDBRAIN recordings (eyes closed and eyes open of the same participant and channel) and compares the coordinates and classifier scores with contiguous processing at the same retained duration.
- `obs_fixedband.py` processes held-out spindle-band simulations in the fixed 11.5-14.5 Hz band of `sleep_checks.py` and in the peak-centred band, and compares the recovery of the fold coordinate.

`direct_fit_baseline.py` compares the simulation-based estimates with the closed-form maximum-likelihood fit of the stationary power density, with and without background deconvolution, on held-out simulations and on the empirical endpoints of `descriptor_benchmark.py`.

## Computation and reproducibility

- Simulation-heavy steps are parallelized with joblib. For these scripts, the last command-line argument is the number of worker processes.
- `run_pipeline.sh` sets `OMP_NUM_THREADS=1` and the equivalent BLAS and Numba variables so that parallel workers do not oversubscribe the CPU.
- Simulations are seeded by their index (chunk number and position within the chunk), so regenerating a chunk reproduces it exactly.
- The alpha posterior is trained on the 189 chunks (94,500 simulations) listed in `config/bank_alpha_chunks.txt`. `sbi_fit.py` restricts the bank to this list when the file is present.
- `run_pipeline.sh` sets `PYTHONHASHSEED=0`, which makes the per-file IAAFT seeds of `run_nf_tdbrain.py` repeatable across runs.
- Bootstrap and permutation procedures use fixed seeds.
- The simulation banks, the twin-paired mechanism banks and the ANPHY-Sleep summary statistics dominate the run time. A multi-core workstation is recommended.
