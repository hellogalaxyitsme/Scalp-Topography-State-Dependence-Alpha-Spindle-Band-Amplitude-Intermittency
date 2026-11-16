#!/bin/bash
# Complete sequence of analysis steps, in order. Run from the repository root after `pip install -e .`.
# The last argument of the parallel steps is the number of worker processes; adjust it to the machine.
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONHASHSEED=0
mkdir -p results
NJ=${NJ:-8}

# 1. Simulation banks (500 simulations per chunk) and amortized posteriors
python3 scripts/sbi_bank.py alpha    0 200 $NJ            # alpha bank; the posterior is trained on the chunks in config/bank_alpha_chunks.txt
python3 scripts/sbi_bank.py sigma250 0 80  $NJ            # spindle bank, 40,000 simulations
python3 scripts/sbi_bank.py alpha    0 20  $NJ mod        # slow (3-30 s) hidden modulation, 10,000 simulations
python3 scripts/sbi_bank.py alpha    0 40  $NJ fastmod    # fast (0.2-5 s) modulation of the Hopf parameter
python3 scripts/sbi_bank.py sigma250 0 40  $NJ fastmod
python3 scripts/sbi_fit.py alpha full
python3 scripts/sbi_fit.py sigma250 full
python3 scripts/eval_mod.py alpha

# 2. Empirical summary statistics
python3 scripts/run_chaos_tdbrain.py $NJ                   # spectral covariates (and complexity indices)
python3 scripts/run_nf_tdbrain.py $NJ
python3 scripts/run_nf_tdbrain_variants.py eog $NJ
python3 scripts/run_nf_tdbrain_variants.py csd $NJ
python3 scripts/run_nf_sleep.py $NJ
python3 scripts/run_mumtaz.py $NJ
python3 scripts/run_ds003478.py $NJ
python3 scripts/run_nf_anphy.py prep 2                     # one-off conversion of ANPHY-Sleep nights to the cache
python3 scripts/run_nf_anphy.py summ $NJ
python3 scripts/run_nf_anphy.py summ $NJ csd

# 3. Posteriors and simulated support (input directory, output outside it)
python3 scripts/nf_apply.py results/sbi_alpha_full.joblib    results/nf_tdbrain     results/post_tdbrain.csv
python3 scripts/nf_apply.py results/sbi_alpha_full.joblib    results/nf_tdbrain_eog results/post_tdbrain_eog.csv
python3 scripts/nf_apply.py results/sbi_alpha_full.joblib    results/nf_tdbrain_csd results/post_tdbrain_csd.csv
python3 scripts/nf_apply.py results/sbi_alpha_full.joblib    results/mumtaz         results/post_mumtaz.csv kind=nf
python3 scripts/nf_apply.py results/sbi_alpha_full.joblib    results/ds003478       results/post_ds003478.csv kind=nf
python3 scripts/nf_apply.py results/sbi_alpha_full.joblib    results/nf_sleep       results/post_sleep_alpha.csv band=alpha
python3 scripts/nf_apply.py results/sbi_sigma250_full.joblib results/nf_sleep       results/post_sleep_sigma.csv band=sigma250
python3 scripts/nf_apply.py results/sbi_sigma250_full.joblib results/nf_anphy       results/post_anphy_sigma.csv band=sigma250
python3 scripts/nf_apply.py results/sbi_alpha_full.joblib    results/nf_anphy       results/post_anphy_alpha.csv band=alpha
python3 scripts/nf_apply.py results/sbi_sigma250_full.joblib results/nf_anphy_csd   results/post_anphy_csd_sigma.csv band=sigma250
python3 scripts/nf_apply.py results/sbi_alpha_full.joblib    results/nf_anphy_csd   results/post_anphy_csd_alpha.csv band=alpha
python3 scripts/sbi_diagnostics.py
python3 scripts/delta_relations.py
python3 scripts/ood_matched.py
python3 scripts/coverage_rows.py
python3 scripts/coverage_anphy.py
python3 scripts/coverage_anphy.py csd

# 4. Dataset analyses
python3 scripts/analyze_nf_tdbrain.py
python3 scripts/analyze_nf_tdbrain.py --covered
python3 scripts/analyze_sleep.py
python3 scripts/analyze_mumtaz.py
python3 scripts/analyze_ds003478.py
python3 scripts/analyze_anphy.py 5000
python3 scripts/analyze_anphy.py 5000 csd
python3 scripts/fold_stats.py
python3 scripts/fold_calibration.py

# 5. Topography and montage variants
python3 scripts/topo_tdbrain.py 5000
python3 scripts/topo_tdbrain.py 5000 eog
python3 scripts/topo_tdbrain.py 5000 csd
python3 scripts/snr_adjusted_topo.py 5000                 # eye opening at zero SNR change (Freedman-Lane sign flipping)
python3 scripts/variants_summary.py
python3 scripts/peak_sensitivity.py 2000
python3 scripts/twin_checks.py $NJ

# 6. Bursts: posterior predictive check and cross-session prediction
python3 scripts/run_bursts_tdbrain.py $NJ
python3 scripts/run_bursts_tdbrain.py $NJ retest
python3 scripts/ppc_bursts.py $NJ 4
python3 scripts/ppc_bursts.py $NJ 4 retest
python3 scripts/ppc_bursts.py $NJ 16 main ns16            # Monte Carlo stability: 16 realizations (0-3 as above)
python3 scripts/ppc_stability.py
python3 scripts/analyze_bursts.py
python3 scripts/analyze_bursts_retest.py
python3 scripts/descriptor_benchmark.py                    # model coordinates against direct envelope statistics
python3 scripts/direct_fit_baseline.py                     # direct fit of the stationary power density as a baseline

# 7. State axis
python3 scripts/displacements.py
python3 scripts/state_axis.py
python3 scripts/axis_snr_check.py
python3 scripts/sim_alternatives.py $NJ
python3 scripts/axis_perturbations.py
python3 scripts/robustness_axis.py
python3 scripts/state_axis_joint.py                        # joint bootstrap of axis and TDBRAIN references
python3 scripts/axis_covariates.py

# 8. Mechanism
python3 scripts/mech_classifier.py                          # raw-statistic classifier on untransformed statistics (not used for EEG)
python3 scripts/mech_domain.py
for cfg in alpha sigma250; do for cls in intrinsic mult fast; do
  python3 scripts/sbi_twinbank.py $cfg $cls 0 48 $NJ        # 12,000 twin-paired simulations per class and setting
done; done
python3 scripts/mech_twin.py                                # additive-noise intrinsic class vs fast modulation
python3 scripts/mech_sleep_control.py anphy $NJ             # second twins for the sleep negative controls
python3 scripts/mech_sleep_control.py sleep $NJ
python3 scripts/mech_twin_broad.py                          # intrinsic (additive + state-dependent noise) vs fast; TDBRAIN healthy reference sample
python3 scripts/mech_twin_summary.py
python3 scripts/mech_state_checks.py                       # state changes at matched delta, credible peaks, support, SNR
python3 scripts/mech_drivers.py                            # summary-statistic families behind the classifier and domain separation

# 9. Analysis populations
python3 scripts/population_counts.py

# 10. Sensitivity analyses
python3 scripts/tdbrain_checks.py $NJ                      # twins with re-estimated peak and background; DFA
python3 scripts/nf_apply.py results/sbi_alpha_full.joblib    results/tdbrain_checks      results/post_tdbrain_checks.csv
python3 scripts/sleep_checks.py sleep $NJ                  # fixed 11.5-14.5 Hz band; DFA
python3 scripts/nf_apply.py results/sbi_sigma250_full.joblib results/sleep_checks_sleep  results/post_sleep_checks_sleep.csv
python3 scripts/sleep_checks.py anphy $NJ
python3 scripts/nf_apply.py results/sbi_sigma250_full.joblib results/sleep_checks_anphy  results/post_sleep_checks_anphy.csv
python3 scripts/delta_direct.py 2000                       # direct estimator of the fold distance
python3 scripts/ppc_recovery.py $NJ 600 2                  # burst predictions in simulations with known parameters
python3 scripts/ppc_posterior.py $NJ 50                    # burst predictions with posterior draws
python3 scripts/ppc_posterior.py $NJ 50 retest
python3 scripts/sensitivity_checks.py 2000
