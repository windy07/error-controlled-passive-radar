# Error-Controlled Bayesian Localization for Passive Radar With Receiver Calibration

**Xuanrun Liu**  
National Key Laboratory of Radar Signal Processing, School of Electronic Engineering, Xidian University, Xi'an 710071, China  
Contact: 26021110898@stu.xidian.edu.cn

[Read the paper](paper/manuscript.pdf)

This repository contains the core MATLAB implementation and selected numerical data for Bayesian passive-radar localization with frequency-dependent receiver calibration, unequal ambiguity periods, and correlated measurements. The receiver likelihood integrates the calibration parameters and sums the integer ambiguity components. Bounds on the remaining lattice mass and unvisited positions control the finite-grid posterior and its mean.

## Core implementation

The `src` directory contains 22 functions. All distances are in metres unless a function interface specifies sample delays; frequencies are in hertz and SNR values are in decibels.

| Functions | Role |
| --- | --- |
| `passive_geometry`, `passive_geometry_grid` | Bistatic path differences and position derivatives |
| `receiver_gaussian_model`, `receiver_loglikelihood` | Correlated coarse/fine observation model, calibration integration, and source contrasts |
| `wrapped_gaussian_tree` | Conditional expansion of a correlated periodic Gaussian likelihood |
| `wrapped_gaussian_sphere`, `wrapped_gaussian_sphere_batch`, `wrapped_gaussian_theta`, `wrapped_gaussian_box` | Lattice-sum comparison methods |
| `wrapped_gaussian_maximum` | Joint maximum-component comparison |
| `theta_lattice_setup`, `integer_lattice_reduction`, `wrapped_normal_interval`, `wrapped_gaussian_majorant` | Lattice transformations and Gaussian bounds |
| `finite_grid_posterior`, `spatial_grid_posterior_joint` | Posterior mass bounds, position search, and posterior moments |
| `nr_opportunity_burst`, `finite_receiver_channel`, `finite_iq_frontend` | OFDM opportunity signals, receiver channel, and compressed delay measurements |
| `caf_delay_frequency`, `profile_cancelled_caf`, `continuous_range_baseline` | Waveform processing and continuous-position comparisons |

The numerical examples use MATLAB R2024b. Generating a new NR waveform with `nr_opportunity_burst` additionally requires 5G Toolbox. The saved localization example and result inspection require MATLAB only.

## Run one saved localization example

Set the MATLAB current folder to this repository and run:

```matlab
result = demo_localization;
```

The example solves the nominal -10 dB, world-1 observation with the recorded model, grid spacing, observation-derived initialization, and posterior tolerance of `1e-5`. It writes `outputs/localization/localization.mat`. To run again, select a new output directory:

```matlab
result = demo_localization(fullfile(pwd,'outputs','localization_second'));
```

`data/example_input.mat` contains the receiver observations, model, geometry, grid, and initialization. Ground truth and the saved estimate are in `data/example_reference.mat`, which the example loads after inference. The returned comparison reports the difference from the saved posterior mean and the sum of their analytic mean-error allowances. The reference TV bound is approximately `9.93111624e-6`.

## Selected data

| File | Contents |
| --- | --- |
| `factor_evaluation.mat` | 48,600 factor evaluations across dimensions, covariance scales, correlations, tolerances, and input distributions; elapsed times, bounds, and completion indicators |
| `position_computation.mat` | 27,000 position updates across nine configurations, five method/envelope combinations, and three posterior tolerances |
| `measurement_localization.mat` | Seven conditions with 1,000 paired trials each; position estimates, errors, coverage, numerical bounds, and method settings |
| `waveform_localization.mat` | 500 shared waveform worlds; available method/condition outputs, localization errors, success indicators, and receiver settings |
| `grid_refinement.mat` | Paired grid changes and numerical allowances for the 70 grid comparisons |
| `interference_processing.mat` | The normalized profile-CAF comparison on the same 500 waveform worlds |
| `nominal_cdf.mat`, `interference_cdf.mat` | Per-trial errors and CDF arrays for Figs. 7 and 8 |
| `example_input.mat`, `example_reference.mat` | One compact localization case and its reference result |
| `iq_example.mat` | One reference/surveillance IQ record and its delay-search interval |

The six `*_summary.csv` files provide the corresponding aggregate tables. The MAT files retain the trial-level numerical arrays and their original denominators, including unavailable or failed outputs. The profile-CAF comparison uses the existing waveform population and is not an additional independent experiment. The all-pilot and full-record CAF methods use more waveform resources than single-symbol pilot compression, as specified in the paper.

For example:

```matlab
measurement = load(fullfile('data','measurement_localization.mat'));
waveform = readtable(fullfile('data','waveform_summary.csv'));
nominal = load(fullfile('data','nominal_cdf.mat'));
disp(waveform);
```

The released inference kernels correspond to the final waveform implementation. The factor and position timing datasets retain the implementation and execution settings of their original experiments; rerunning the current kernels does not recreate those historical wall-clock measurements.

## Reference

Xuanrun Liu, "Error-Controlled Bayesian Localization for Passive Radar With Receiver Calibration," preprint, 2026.
