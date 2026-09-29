"""
Round 3 supplementary analyses for the FCC co-processing benchmark.

Reproduces the numbers reported in Supplementary Sections S1 to S3.

Public inputs (released in this repository):
    data/fcc_masked_data.csv.zip         un-normalized series on a masked scale, 104,000 samples
    data/normalization_parameters.csv    mean and standard deviation of every masked variable
    results/best_predictions_test.csv    Transformer test predictions (normalized scale)

Every variable in the masked file is a positive linear transformation of the plant
value with undisclosed coefficients. The transformation leaves the z-scores unchanged,
so Sections S1 and S2, which work on the normalized scale, are reproduced exactly from
the public files. Section S3 (renewable LCC) needs the fossil feed, bio feed, and LCC
production in physical units; its benchmark-window part runs only when the
engineering-scale file is supplied with --engineering-data. The published S3 outputs
were produced in that way by the authors and are expressed in dimensionless form.

Usage (from the repository root):
    python analysis/round3_supplementary_analysis.py --scaling
    python analysis/round3_supplementary_analysis.py --scaling --engineering-data PATH

Outputs are written to results/.
"""
import argparse
import os
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")

WINDOW = 10            # sliding-window length used by the benchmark
SEED = 2026            # seed for the block bootstrap

# Renewable-LCC model of Cao et al. (2024), Applied Energy 360:122815, Eqs. (6), (8), (9),
# estimated on the same 104,000-sample dataset.
A_FOSSIL = 0.542       # LCC yield per unit volume of fossil feed
B_BIO = 0.323          # LCC yield per unit volume of bio feed

# Four 14C measurements reported in Cao et al. (2024), Table 3.
C14 = pd.DataFrame({
    "sample": [1, 2, 3, 4],
    "coprocessing_ratio_pct": [10.75, 12.24, 12.32, 12.22],
    "measured_14C_fraction_pct": [6.30, 7.30, 7.80, 7.40],
    "model_14C_fraction_pct": [6.68, 7.65, 7.69, 7.64],
})


def r2(y, f):
    return 1.0 - np.sum((y - f) ** 2) / np.sum((y - y.mean()) ** 2)


def first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def read_csv_any(path):
    """Read a CSV file or the CSV member of a zip archive (ignoring __MACOSX entries)."""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            name = [n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX")][0]
            with z.open(name) as f:
                return pd.read_csv(f)
    return pd.read_csv(path)


def load():
    data = read_csv_any(first_existing(os.path.join(DATA, "fcc_masked_data.csv"),
                                      os.path.join(DATA, "fcc_masked_data.csv.zip")))
    par = pd.read_csv(os.path.join(DATA, "normalization_parameters.csv")).set_index("Variable")
    pred = pd.read_csv(os.path.join(RES, "best_predictions_test.csv"))
    return data, par, pred


def test_rows(n_total, n_test):
    """Row indices of the test targets under the benchmark split."""
    n_seq = n_total - WINDOW
    val_end = int(n_seq * 0.9)
    idx = np.arange(val_end, n_seq) + WINDOW
    assert len(idx) == n_test, (len(idx), n_test)
    return idx


# ------------------------------------------------------------------ S1
def s1_consistency(data, par):
    """Check that normalization_parameters.csv maps the masked file onto fcc_normalized_data.csv."""
    path = first_existing(os.path.join(DATA, "fcc_normalized_data.csv"),
                          os.path.join(ROOT, "fcc_normalized_data.csv"),
                          os.path.join(ROOT, "fcc_normalized_data.csv.zip"))
    if path is None:
        print("S1: fcc_normalized_data.csv not found, check skipped")
        return
    norm = read_csv_any(path)
    err = max(np.max(np.abs((data[c] - par.loc[c, "Mean"]) / par.loc[c, "Std"] - norm[c]))
              for c in par.index)
    print(f"S1: max |(masked - Mean)/Std - normalized| = {err:.2e}")


# ------------------------------------------------------------------ S2
def s2_error_analysis(data, par, pred):
    """Error analysis of the Transformer on the normalized scale (units of the target standard deviation)."""
    idx = test_rows(len(data), len(pred))
    z_check = (data["Target"].values[idx] - par.loc["Target", "Mean"]) / par.loc["Target", "Std"]
    assert np.allclose(z_check, pred["Actual"].values), "prediction file not aligned with the data file"
    y = pred["Actual"].values
    f = pred["Predicted"].values
    e = y - f                                   # positive = under-prediction
    mse = np.mean(e ** 2)

    overall = pd.DataFrame([{
        "n_test": len(y), "R2": r2(y, f), "N_RMSE": np.sqrt(mse), "N_MAE": np.mean(np.abs(e)),
        "mean_error_sigma": e.mean(),
    }])

    # Theil decomposition of the MSE
    rho = np.corrcoef(y, f)[0, 1]
    theil = pd.DataFrame([{
        "bias_proportion_UM": (f.mean() - y.mean()) ** 2 / mse,
        "variance_proportion_US": (f.std() - y.std()) ** 2 / mse,
        "covariance_proportion_UC": 2 * (1 - rho) * f.std() * y.std() / mse,
        "correlation": rho, "std_actual_sigma": y.std(), "std_predicted_sigma": f.std(),
    }])

    # heteroscedasticity: quintiles of the actual value
    q = np.quantile(y, [0, .2, .4, .6, .8, 1])
    rows = []
    for k in range(5):
        m = (y >= q[k]) & ((y < q[k + 1]) if k < 4 else (y <= q[k + 1]))
        ek = e[m]
        rows.append({"bin": f"Q{k+1}", "lower_sigma": q[k], "upper_sigma": q[k + 1], "n": int(m.sum()),
                     "mean_error_sigma": ek.mean(), "N_RMSE": np.sqrt(np.mean(ek ** 2))})
    bins = pd.DataFrame(rows)

    # bias correction, residual autocorrelation, moving-block bootstrap
    fc = f + e.mean()
    r0 = e - e.mean()
    acf1 = np.sum(r0[:-1] * r0[1:]) / np.sum(r0 ** 2)
    rng = np.random.default_rng(SEED)
    n = len(y)
    boot = {}
    for L in (144, 1008):                       # 1 day and 7 days at 10-min sampling
        nb = int(np.ceil(n / L))
        vals = []
        for _ in range(1000):
            st = rng.integers(0, n - L + 1, nb)
            ii = np.concatenate([np.arange(s, s + L) for s in st])[:n]
            yy, ff = y[ii], f[ii]
            vals.append(r2(yy, ff + np.mean(yy - ff)))
        boot[L] = np.percentile(vals, [2.5, 97.5])
    bias = pd.DataFrame([{
        "R2": r2(y, f), "constant_offset_sigma": e.mean(), "bias_corrected_R2": r2(y, fc),
        "residual_ACF1": acf1,
        "CI95_block1day_low": boot[144][0], "CI95_block1day_high": boot[144][1],
        "CI95_block7day_low": boot[1008][0], "CI95_block7day_high": boot[1008][1],
    }])

    overall.to_csv(os.path.join(RES, "S2_overall_metrics.csv"), index=False)
    theil.to_csv(os.path.join(RES, "S2_error_decomposition.csv"), index=False)
    bins.to_csv(os.path.join(RES, "S2_heteroscedasticity_bins.csv"), index=False)
    bias.to_csv(os.path.join(RES, "S2_bias_correction.csv"), index=False)
    print("S2 overall\n", overall.round(4).to_string(index=False))
    print("S2 Theil\n", theil.round(4).to_string(index=False))
    print("S2 bins\n", bins.round(3).to_string(index=False))
    print("S2 bias\n", bias.round(4).to_string(index=False))


# ------------------------------------------------------------------ S3
def renewable_lcc(F, B, L):
    """Eq. (8) of Cao et al. (2024): renewable LCC = b*B + r*(LCC - a*F - b*B)."""
    r = B / (F + B)
    return B_BIO * B + r * (L - A_FOSSIL * F - B_BIO * B), r


def s3_14c():
    """Check of the fraction model against the four published 14C measurements."""
    c = C14.copy()
    c["error_pp"] = c.model_14C_fraction_pct - c.measured_14C_fraction_pct
    s, i = np.polyfit(c.coprocessing_ratio_pct, c.measured_14C_fraction_pct, 1)
    p = s * c.coprocessing_ratio_pct + i
    summary = pd.DataFrame([{
        "n": len(c), "slope_pp_per_pp": s, "intercept_pp": i,
        "R2": r2(c.measured_14C_fraction_pct.values, p.values),
        "mean_measured_pct": c.measured_14C_fraction_pct.mean(),
        "CV_measured_pct": 100 * c.measured_14C_fraction_pct.std(ddof=1) / c.measured_14C_fraction_pct.mean(),
        "model_MAE_pp": np.mean(np.abs(c.error_pp)), "model_bias_pp": c.error_pp.mean(),
        "model_MAPE_pct": 100 * np.mean(np.abs(c.error_pp) / c.measured_14C_fraction_pct),
    }])
    c.to_csv(os.path.join(RES, "S3_14C_validation.csv"), index=False)
    summary.to_csv(os.path.join(RES, "S3_14C_regression_summary.csv"), index=False)
    print("S3 14C\n", summary.round(3).to_string(index=False))


def s3_window(eng, pred):
    """Renewable LCC over the benchmark window; needs the engineering-scale feeds and LCC (bbl/day)."""
    F, B, L = eng["Feature_01"].values, eng["Feature_02"].values, eng["Target"].values
    RL, r = renewable_lcc(F, B, L)
    f = 100 * RL / L                                   # Eq. (9), renewable fraction in %

    # regression of renewable LCC on total LCC, both relative to their window means,
    # so that strict proportionality would give slope 1 and intercept 0
    x, yv = L / L.mean(), RL / RL.mean()
    sl, il = np.polyfit(x, yv, 1)
    sf, if_ = np.polyfit(100 * r, f, 1)
    act = r >= 0.01
    m = (RL > 0) & act
    lf, lL = np.log(f[m]), np.log(L[m])
    v = np.var(lf + lL)
    bench = pd.DataFrame([{
        "coprocessing_ratio_mean_pct": 100 * r.mean(), "coprocessing_ratio_sd_pct": 100 * r.std(),
        "coprocessing_ratio_max_pct": 100 * r.max(), "share_time_ratio_below_1pct": 100 * np.mean(~act),
        "fraction_mean_pct": f.mean(), "fraction_sd_pct": f.std(), "fraction_CV_pct": 100 * f.std() / f.mean(),
        "fraction_max_pct": f.max(),
        "fraction_mean_when_coprocessing_pct": f[act].mean(),
        "fraction_CV_when_coprocessing_pct": 100 * f[act].std() / f[act].mean(),
        "renewable_LCC_CV_pct": 100 * RL.std() / RL.mean(),
        "reg_renewable_on_total_rel_slope": sl, "reg_renewable_on_total_rel_intercept": il,
        "reg_renewable_on_total_R2": r2(yv, sl * x + il),
        "reg_fraction_on_ratio_slope": sf, "reg_fraction_on_ratio_intercept_pp": if_,
        "reg_fraction_on_ratio_R2": r2(f, sf * 100 * r + if_),
        "var_share_log_renewable_from_fraction_pct": 100 * np.var(lf) / v,
        "var_share_log_renewable_from_totalLCC_pct": 100 * np.var(lL) / v,
        "var_share_log_renewable_covariance_pct": 100 * 2 * np.cov(lf, lL, ddof=0)[0, 1] / v,
    }])

    # renewable-carbon estimators on the benchmark test window; the Transformer prediction
    # is mapped back with the full-series statistics, as in code.py
    mu, sd = L.mean(), L.std()
    idx = test_rows(len(eng), len(pred))
    assert np.allclose((L[idx] - mu) / sd, pred["Actual"].values), "prediction file not aligned"
    Lp = pred["Predicted"].values * sd + mu
    ref = RL[idx]
    two_stage, _ = renewable_lcc(F[idx], B[idx], Lp)
    n_train_rows = int((len(eng) - WINDOW) * 0.8) + WINDOW
    fbar = RL[:n_train_rows].sum() / L[:n_train_rows].sum()
    const = fbar * L[idx]
    rows = []
    for name, est in [("Two-stage: Transformer LCC soft sensor + Eq. (8)", two_stage),
                      (f"Constant fraction ({100*fbar:.2f}% from training period) x measured LCC", const)]:
        e = ref - est
        rows.append({"estimator": name, "R2": r2(ref, est),
                     "RMSE_pct_of_mean": 100 * np.sqrt(np.mean(e ** 2)) / ref.mean(),
                     "MAE_pct_of_mean": 100 * np.mean(np.abs(e)) / ref.mean(),
                     "mean_error_pct_of_mean": 100 * e.mean() / ref.mean()})
    base = pd.DataFrame(rows)

    bench.to_csv(os.path.join(RES, "S3_renewable_fraction_benchmark.csv"), index=False)
    base.to_csv(os.path.join(RES, "S3_renewable_carbon_baselines.csv"), index=False)
    print("S3 benchmark window\n", bench.round(3).T.to_string(header=False))
    print("S3 baselines\n", base.round(4).to_string(index=False))
    return RL, r, f


# ------------------------------------------------------------------ scaling check
def scaling_check(data):
    """Test R2 of the linear baselines with full-series versus training-only z-score statistics."""
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    import warnings
    warnings.filterwarnings("ignore")
    X = data[[f"Feature_{i:02d}" for i in range(1, 16)]].values.astype(float)
    y = data["Target"].values.astype(float)
    n = len(X) - WINDOW
    tr, va = int(n * 0.8), int(n * 0.9)
    out = []
    for mode in ("full series", "training only"):
        cut = len(X) if mode == "full series" else tr + WINDOW
        mx, sx, my, sy = X[:cut].mean(0), X[:cut].std(0), y[:cut].mean(), y[:cut].std()
        Xs, ys = (X - mx) / sx, (y - my) / sy
        Xf = np.stack([Xs[i:i + WINDOW].ravel() for i in range(n)])
        yf = ys[WINDOW:WINDOW + n]
        for name, mdl in [("OLS", LinearRegression()), ("Ridge", Ridge(alpha=1.0)),
                          ("Lasso", Lasso(alpha=0.01)), ("ElasticNet", ElasticNet(alpha=0.01, l1_ratio=0.5))]:
            mdl.fit(Xf[:tr], yf[:tr])
            pr = mdl.predict(Xf[va:])
            out.append({"model": name, "scaling": mode, "test_R2": r2(yf[va:], pr)})
    df = pd.DataFrame(out).pivot(index="model", columns="scaling", values="test_R2")
    df["difference"] = df["training only"] - df["full series"]
    df.to_csv(os.path.join(RES, "S1_scaling_sensitivity.csv"))
    print("Scaling check\n", df.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scaling", action="store_true", help="also run the normalization-scope check")
    ap.add_argument("--engineering-data", default=None,
                    help="engineering-scale file (Target, Feature_01, Feature_02 in bbl/day); "
                         "not public, needed only for the benchmark-window part of Section S3")
    args = ap.parse_args()
    data, par, pred = load()
    s1_consistency(data, par)
    s2_error_analysis(data, par, pred)
    s3_14c()
    if args.engineering_data:
        s3_window(read_csv_any(args.engineering_data), pred)
    else:
        print("S3: benchmark-window analysis skipped (needs --engineering-data); "
              "the released S3_renewable_*.csv files were produced by the authors from the engineering-scale records")
    if args.scaling:
        scaling_check(data)
