# Data-Driven Maintenance Optimization: Aligning Prognostic Learning with Decision Objective

Research code for remaining useful life (RUL) prediction and maintenance decision-making on the NASA CMAPSS datasets FD001–FD004. The project compares **Estimate-then-Optimize (ETO)**, **Decision-Oriented hyperparameter Tuning (DOT)**, and **Integrated Estimate-Optimize (IEO)** using a shared LSTM architecture with Monte Carlo dropout.

## Environment

The GPU setup targets the following environment:

| Component | Version |
| --- | --- |
| Python environment | `py312` |
| Python | 3.12 |
| PyTorch | 2.11.0 |
| CUDA build | 12.8 (`cu128`) |


For the GPU setup, the output should identify PyTorch `2.11.0+cu128`, CUDA `12.8`, and GPU availability as `True`.

## Run training

Run the following command from the project root:

```sh
python main.py --machine FD001 --factor 5 --trials 50
```

This runs all three approaches, in the order ETO, DOT, and IEO, for 50 trials using data-split seeds 0–49. IEO is initialized from the ETO model trained in the same trial.

| Argument | Description |
| --- | --- |
| `--machine` | Dataset: `FD001`, `FD002`, `FD003`, or `FD004` |
| `--factor` | Failure-to-opportunity cost ratio, `c_f / c_p`; experiment values are `2`, `5`, `10`, and `20` |
| `--trials` | Number of repeated trials; use `50` for the command above |

All three arguments must be supplied. Holding cost is fixed at `c_h = 1`, opportunity cost at `c_p = 5`, and failure cost is `c_f = factor * c_p`. Thus, `--factor 5` gives `c_f = 25`.

**Device selection:** `main.py` currently sets `device = "cpu"`. To train on the GPU, change that assignment near the top of the file to:

```python
device = "cuda:0"
```

Each trial performs a hyperparameter search for each approach, so a 50-trial run can take substantial time. Results are saved after all trials finish to `results/FD001_ratio=05.pth` for the example above. Keep the `results/` directory present. Running the same dataset and ratio again overwrites its result file; preserve existing results before rerunning.

## Training and validation criteria

All approaches use the same LSTM architecture. Their training objectives and validation criteria differ as follows:

| Approach | Training criterion | Validation criterion |
| --- | --- | --- |
| **ETO — Estimate-then-Optimize** | Mean squared error (MSE) between predicted and observed RUL | MSE averaged over 1,000 floored Monte Carlo RUL samples per validation observation |
| **DOT — Decision-Oriented hyperparameter Tuning** | MSE between predicted and observed RUL | Average maintenance cost of decisions obtained from the predicted RUL distribution, evaluated against observed validation RUL |
| **IEO — Integrated Estimate-Optimize** | A finite-difference surrogate for the decision-cost gradient, starting from ETO weights | Average maintenance cost, using the same validation procedure as DOT |

For each approach, its validation criterion controls both **early stopping** and **hyperparameter selection**. Lower scores are preferred. ETO and DOT use the same prediction loss during training; DOT introduces decision cost when selecting the stopping epoch and hyperparameters. IEO also incorporates decision cost into the training updates.

In IEO, the surrogate gradient is computed by comparing decisions associated with neighboring RUL support points and evaluating their costs against observed RUL. The training loss is `mean(predicted_RUL * surrogate_gradient)`, with the gradient treated as a fixed quantity during backpropagation.

Validation and check-set predictions use 1,000 Monte Carlo dropout samples per observation. For decision optimization, samples are floored and mapped to the discrete RUL support `0, ..., 124`. Maintenance costs are averaged over equally likely lead times of 3–7 cycles.

Both optimization routines use the feasible set:

```text
z_o <= z_r
z_d <= z_r
```

Here, `z_o`, `z_d`, and `z_r` are the ordering, downtime, and replacement times. Ordering and downtime times can each range from 0 to 99; replacement times are 5, 10, ..., 100.

## Experimental settings

| Setting | Value |
| --- | --- |
| Input window length | 30 cycles |
| LSTM hidden units / dense hidden units | 50 / 100 |
| Dropout probability | 0.20 |
| Optimizer / batch size | Adam / 100 |
| Learning-rate candidates | `1e-4`, `2e-4`, `5e-4`, `1e-3` |
| Maximum epochs per candidate | 1,000 |
| Early-stopping patience | 10 consecutive epochs without validation improvement |

Prepared data are included as `data/X_Y_FD001.csv` through `data/X_Y_FD004.csv`. Each file contains engine identifiers, 21 sensor measurements, and RUL labels derived from run-to-failure trajectories. Dataset information is available from the [NASA Prognostics Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/).

The loader uses the first 100 engine groups from each dataset: 70 are randomly selected for training, the first 10 remaining groups in index order form validation, and the final 20 form the check set. It retains windows with target RUL at most 99, removes constant or nearly constant sensors, and normalizes features using statistics from the final time steps of training windows. The check set therefore comes from held-out run-to-failure engines rather than the official C-MAPSS test files.

Trial indices seed the NumPy data split. PyTorch initialization, batch shuffling, and dropout are not explicitly seeded, so repeated runs may produce different numerical results.

## Results and analysis

Result files contain metric tensors of shape `[trials, 3, 7]`, with methods ordered **ETO, DOT, IEO**. A 50-trial run produces a tensor of shape `[50, 3, 7]`.

| Metric index | Quantity |
| --- | --- |
| 0 | MAE averaged across observations and Monte Carlo samples |
| 1 | Left CRPS component |
| 2 | Right CRPS component |
| 3 | Average maintenance cost |
| 4 | Fraction of decisions with `z_d <= observed RUL` |
| 5 | Fraction of decisions with `z_d > observed RUL` |
| 6 | Training and tuning time in seconds, excluding check-set evaluation |

Total CRPS is the sum of components 1 and 2. IEO's stored time covers its own tuning stage; add ETO's time to account for its prerequisite training. The files store experiment metrics, not model weights.

The main project files are:

| Path | Purpose |
| --- | --- |
| `main.py` | Experiment entry point |
| `methods/ETO.py`, `methods/DOT.py`, `methods/IEO.py` | Training and validation implementations |
| `opt.py` | Decision optimization, cost evaluation, and surrogate gradients |
| `utlis.py` | Data loading, LSTM architecture, Monte Carlo prediction, and CRPS |
| `data/preprocess.ipynb` | Generate prepared CSV files from raw trajectories; run from `data/` |
| `postanalysis/experiments.ipynb` | Analyze saved results, generate comparison figures and tables, and run statistical tests |
| `postanalysis/example1.ipynb`, `postanalysis/proposition1.ipynb` | Illustrative examples and figures |

Run analysis notebooks with `postanalysis/` as their working directory. Figures are saved in `postanalysis/figs/`. Some plotting cells require LaTeX and request CMU Serif fonts to reproduce the supplied styling.
