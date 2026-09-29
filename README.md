# CEP-FCC-Benchmark

Real-World Industrial Process Monitoring Benchmark for Bio-feedstock Co-processing in Fluid Catalytic Cracking Units

This repository contains the dataset, benchmark code, results, and supplementary analyses for the benchmark described in the paper submitted to *Control Engineering Practice* (manuscript CONENGPRAC-D-26-00273). The data come from a commercial FCC unit at Parkland Refining (B.C.) Ltd. and cover about three years of co-processing operation at a 10-minute sampling interval.

The version referred to in the third-round response to reviewers is the tagged release **v1.1-round3**.

## Repository layout

| Path | Content |
|---|---|
| `fcc_normalized_data.csv.zip` | z-scored benchmark series: 104,000 samples, the target and 15 features |
| `data/fcc_masked_data.csv.zip` | the same 104,000 samples before z-scoring, on a masked scale (see below) |
| `data/normalization_parameters.csv` | mean and standard deviation of every masked variable, with its physical quantity and recording unit |
| `code.py` | benchmark code: reads the masked file, normalizes it, trains and evaluates the 30 models |
| `results/benchmark_results_27_models.csv` | train and test metrics of the 27 reported baselines |
| `results/excluded_models_negative_test_R2.csv` | the three evaluated baselines with negative test R² |
| `results/best_predictions_test.csv` | Transformer test predictions (normalized scale) |
| `analysis/round3_supplementary_analysis.py` | produces the numbers in Supplementary Sections S1 to S3 |
| `results/S1_*.csv`, `results/S2_*.csv`, `results/S3_*.csv` | outputs of the analysis script |
| `supplementary/Supplementary_Material_Round3.pdf` | Supplementary Material of the revised manuscript |

## Data

Each row is one 10-minute sample, in chronological order. Absolute timestamps are replaced by a sequential index; the order and spacing of the samples are unchanged.

**Masking.** The absolute operating values of the unit are commercially confidential. Every variable, including the target, is therefore released on a masked scale: the masked value is a positive linear transformation of the plant value, `x_masked = s * x + o` with `s > 0`, and the coefficients differ between variables and are not disclosed. A positive linear transformation leaves the z-score of every sample unchanged, so the masked file and the normalized file lead to identical benchmark results. The units below are the units in which the plant records each variable; they describe the physical meaning of the variable, not the scale of the released values.

| Column | Physical quantity | Recording unit |
|---|---|---|
| Target | Total LCC (light catalytic cracked oil) production | bbl/day |
| Feature_01 | Fossil feed flow | bbl/day |
| Feature_02 | Bio-feed flow | bbl/day |
| Feature_03 | Outlet flow 1 | kbbl/day |
| Feature_04 | Outlet flow 2 | kbbl/day |
| Feature_05 | Catalyst cooler fluffing air | kSCF/h (thousand standard ft³ per hour) |
| Feature_06 | Catalyst circulation rate | short ton/min |
| Feature_07 | Flue gas excess oxygen | vol% |
| Feature_08 | CO concentration signal | ppm |
| Feature_09 | Riser temperature | °F |
| Feature_10 | Preheat temperature | °F |
| Feature_11 | Catalyst cooler steam flow | klb/h (thousand lb per hour) |
| Feature_12 | Upper regenerator temperature | °F |
| Feature_13 | Catalyst to oil ratio | dimensionless |
| Feature_14 | Reactor temperature | °F |
| Feature_15 | Conversion rate during co-processing | % |

**Normalization.** `data/normalization_parameters.csv` gives the mean and standard deviation of each masked variable over the full series, which is how `code.py` applies the z-score transform. Computing `(masked - Mean) / Std` reproduces `fcc_normalized_data.csv` to within 2e-12. Refitting the linear baselines with training-period statistics changes their test R² by at most 0.0009 (`results/S1_scaling_sensitivity.csv`).

**Split.** Sliding windows of 10 steps are built first, and the resulting sequences are split 80/10/10 in time order. The test targets are rows 93,601 to 103,999 of the data files.

## Quick start

```bash
pip install -r requirements.txt

# Reproduce the benchmark (the sklearn baselines run on CPU; the deep models benefit from a GPU)
python code.py                                   # reads data/fcc_masked_data.csv.zip

# Reproduce Supplementary Sections S1, S2 and Table S8 (CPU, under a minute)
python analysis/round3_supplementary_analysis.py --scaling
```

On the masked file, `code.py` reproduces the test R² of the ten deterministic linear baselines in Table 6 of the paper to four decimal places.

## Reported and excluded baselines

`code.py` evaluates 30 models. Three of them (PassiveAggressive, RANSAC, Lars) have a negative test R², meaning that they perform worse than predicting the mean, and they are not included among the 27 baselines reported in the paper. Their metrics are released in `results/excluded_models_negative_test_R2.csv`.

## Renewable LCC

The renewable portion of LCC is computed with Eq. (8) of Cao et al. (2024), which was estimated on this dataset and checked against the four ¹⁴C measurements reported in that paper (Table 3). The model coefficients and the four measurements are written into the analysis script together with their source, and the ¹⁴C check (`results/S3_14C_*.csv`) runs from the public files.

The benchmark-window part of the analysis needs the fossil feed, the bio feed, and LCC production in physical units, which the masked release does not contain. The files `results/S3_renewable_*.csv` were produced by the authors with the same script from the engineering-scale records (`--engineering-data`), and they report dimensionless quantities only.

## Citation

If you use this benchmark, please cite the paper (details will be updated on acceptance):

```
L. Cao, J. Su, L. C. Siang, Y. Wang, Y. Cao, G. Lee, J. Li, R. B. Gopaluni.
Real-World Industrial Process Monitoring Benchmark for Bio-feedstock Co-processing
in Fluid Catalytic Cracking Units. Control Engineering Practice (under review).
```

and the study from which the renewable-LCC model is taken:

```
L. Cao, J. Su, J. Saddler, Y. Cao, Y. Wang, G. Lee, L. C. Siang, R. Pinchuk, J. Li, R. B. Gopaluni.
Real-time tracking of renewable carbon content with AI-aided approaches during co-processing
of biofeedstocks. Applied Energy 360 (2024) 122815.
```

## License

Released under the MIT License (see `LICENSE`).

## Contact

R. Bhushan Gopaluni, bhushan.gopaluni@ubc.ca
