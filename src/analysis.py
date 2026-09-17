"""Reproduce all statistical outputs from the fixed price snapshot.

PCA uses the unbiased sample covariance. Mardia uses the ML covariance (1/n).
No normality assumption is required for the descriptive PCA decomposition.
"""
from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from sklearn.decomposition import FactorAnalysis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PERIODS = {
    "full": ("Full study", "2023-01-01", "2025-12-31"),
    "comparison": ("Comparison quarter", "2024-07-01", "2024-09-30"),
    "stress": ("Stress candidate quarter", "2024-10-01", "2024-12-31"),
}

def load_returns():
    prices = pd.read_csv(ROOT / "data/adjusted_prices.csv", index_col=0, parse_dates=True)
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Price dates must be unique and increasing.")
    if (prices <= 0).any().any():
        raise ValueError("Prices must be positive.")
    # Calculate before removing incomplete rows: do not bridge gaps silently.
    all_returns = np.log(prices / prices.shift(1))
    valid = all_returns.dropna()
    if not np.isfinite(valid.to_numpy()).all():
        raise ValueError("Non-finite returns found.")
    diagnostics = {"price_rows": len(prices), "return_rows": len(valid),
                   "incomplete_return_rows_excluding_first": int(all_returns.iloc[1:].isna().any(axis=1).sum()),
                   "start": str(valid.index[0].date()), "end": str(valid.index[-1].date()),
                   "missing_prices": int(prices.isna().sum().sum())}
    return valid.drop(columns="^NSEI"), valid["^NSEI"], diagnostics

def pca_covariance(x):
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    if n <= p or not np.isfinite(x).all():
        raise ValueError("PCA needs finite data and more observations than variables.")
    cov = np.cov(x, rowvar=False, ddof=1)
    eigenvalues, vectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    if eigenvalues[-1] <= 1e-12 or eigenvalues[0] / eigenvalues[-1] > 1e10:
        raise ValueError("Covariance is singular or ill-conditioned; remove duplicate/constant series.")
    # Deterministic signs for reproducible plots; the signs have no risk meaning.
    for k in range(p):
        if vectors[np.argmax(np.abs(vectors[:, k])), k] < 0:
            vectors[:, k] *= -1
    w = np.ones(p) / p
    variance = float(w @ cov @ w)
    contributions = eigenvalues * (w @ vectors) ** 2
    assert np.isclose(contributions.sum(), variance, rtol=1e-10)
    assert np.allclose(vectors @ np.diag(eigenvalues) @ vectors.T, cov)
    loadings = vectors * np.sqrt(eigenvalues)[None, :] / np.sqrt(np.diag(cov))[:, None]
    return cov, eigenvalues, vectors, loadings, variance, contributions

def mardia(x):
    x = np.asarray(x)
    n, p = x.shape
    centered = x - x.mean(axis=0)
    cov_ml = centered.T @ centered / n
    gram = centered @ np.linalg.solve(cov_ml, centered.T)
    b1 = float(np.mean(gram ** 3))
    b2 = float(np.mean(np.diag(gram) ** 2))
    skew_stat = n * b1 / 6
    skew_df = p * (p + 1) * (p + 2) // 6
    kurt_z = (b2 - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
    ps, pk = float(chi2.sf(skew_stat, skew_df)), float(2 * norm.sf(abs(kurt_z)))
    return {"b1": b1, "b2": b2, "skew_chi2": skew_stat, "skew_df": skew_df,
            "skew_p": ps, "kurt_z": float(kurt_z), "kurt_p": pk,
            "reject_at_5pct_bonferroni": bool(min(ps, pk) < 0.025)}

def period_analysis(x, benchmark, key):
    label, start, end = PERIODS[key]
    x = x.loc[start:end]
    benchmark = benchmark.loc[x.index]
    cov, eig, vec, load, variance, cont = pca_covariance(x)
    p = x.shape[1]
    w = np.ones(p) / p
    # Exact daily return for a portfolio reset to equal value weights each day.
    portfolio_simple = np.expm1(x.to_numpy()) @ w
    loss = -portfolio_simple
    hist_var = float(np.quantile(loss, .95, method="linear"))
    corr = x.corr().to_numpy()
    joint = np.cov(np.column_stack([x, benchmark]), rowvar=False, ddof=1)
    residual_cov = cov - np.outer(joint[:-1, -1], joint[:-1, -1]) / joint[-1, -1]
    logdet = float(np.linalg.slogdet(cov)[1])
    scores = (x.to_numpy() - x.mean().to_numpy()) @ vec
    benchmark_correlation = float(np.corrcoef(scores[:, 0], benchmark)[0, 1])
    return {"key": key, "label": label, "n": len(x), "start": str(x.index[0].date()),
            "end": str(x.index[-1].date()), "tickers": [c.replace(".NS", "") for c in x.columns],
            "covariance": cov.tolist(), "correlation": corr.tolist(), "eigenvalues": eig.tolist(),
            "eigenvectors": vec.tolist(), "loadings": load.tolist(),
            "explained_pct": (100 * eig / eig.sum()).tolist(),
            "portfolio_risk_pct": (100 * cont / variance).tolist(),
            "portfolio_variance": variance, "daily_volatility_pct": 100 * np.sqrt(variance),
            "mean_correlation": float(corr[np.triu_indices(p, 1)].mean()),
            "historical_var95_pct": 100 * hist_var,
            "log_generalised_variance": logdet, "generalised_variance": float(np.exp(logdet)),
            "residual_portfolio_variance": float(w @ residual_cov @ w),
            "market_linear_variance_share_pct": float(100 * (1 - (w @ residual_cov @ w) / variance)),
            "pc1_nifty_correlation": benchmark_correlation,
            "benchmark_period_return_pct": float(100 * np.expm1(benchmark.sum())),
            "portfolio_period_return_pct": float(100 * (np.prod(1 + portfolio_simple) - 1)),
            "dates": [str(d.date()) for d in x.index], "loss_pct": (100 * loss).tolist()}

def factor_analysis(x):
    # Three factors fixed in advance for a compact exploratory model, not tuned to results.
    a = x.to_numpy()
    scales = a.std(axis=0, ddof=0)
    z = (a - a.mean(axis=0)) / scales
    model = FactorAnalysis(n_components=3, rotation="varimax", svd_method="lapack", max_iter=3000)
    with warnings.catch_warnings(record=True) as messages:
        model.fit(z)
    if any("converge" in str(m.message).lower() for m in messages):
        raise RuntimeError("Factor analysis did not converge.")
    loads = model.components_.T
    for k in range(3):
        if loads[np.argmax(np.abs(loads[:, k])), k] < 0:
            loads[:, k] *= -1
    B = loads * scales[:, None]
    psi = model.noise_variance_ * scales ** 2
    w = np.ones(a.shape[1]) / a.shape[1]
    shared = float(np.sum((w @ B) ** 2))
    specific = float(np.sum(w ** 2 * psi))
    residual = np.cov(z, rowvar=False, ddof=0) - (loads @ loads.T + np.diag(model.noise_variance_))
    return {"factors": 3, "loadings_standardised": loads.tolist(),
            "specific_variances_standardised": model.noise_variance_.tolist(),
            "shared_portfolio_variance": shared, "specific_portfolio_variance": specific,
            "shared_share_pct": 100 * shared / (shared + specific),
            "specific_share_pct": 100 * specific / (shared + specific),
            "covariance_rmse_standardised": float(np.sqrt(np.mean(residual ** 2))),
            "iterations": int(model.n_iter_)}

def plots(results, stocks):
    out = ROOT / "outputs"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 200})
    full = results["periods"]["full"]
    fig, ax = plt.subplots(figsize=(7.4, 3.5), layout="constrained")
    ax.bar(range(1, 11), full["explained_pct"], color="#245b78", label="Individual component")
    ax.plot(range(1, 11), np.cumsum(full["explained_pct"]), "o-", color="#b65b28", label="Cumulative")
    ax.set(xlabel="Principal component", ylabel="Explained variance (%)", xticks=range(1, 11), ylim=(0, 105))
    ax.legend(frameon=False)
    ax.set_title("Variation across ten stock returns", loc="left", weight="bold")
    fig.savefig(out / "scree_plot.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4, 4.0), layout="constrained")
    im = ax.imshow(np.array(full["loadings"])[:, :3], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set(yticks=range(10), yticklabels=full["tickers"], xticks=range(3), xticklabels=["PC1", "PC2", "PC3"], xlabel="Principal component", ylabel="NSE stock")
    for i in range(10):
        for j in range(3):
            v = full["loadings"][i][j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if abs(v) > .6 else "black")
    fig.colorbar(im, ax=ax, label="Correlation of stock return with component")
    ax.set_title("PCA loadings in the full study", loc="left", weight="bold")
    fig.savefig(out / "loadings_heatmap.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4, 3.4), layout="constrained")
    ax.bar(range(1, 11), full["portfolio_risk_pct"], color="#245b78")
    ax.set(xlabel="Principal component", ylabel="Share of portfolio variance (%)", xticks=range(1, 11))
    ax.set_title("Equal weight portfolio risk contributions", loc="left", weight="bold")
    fig.savefig(out / "risk_contributions.png"); plt.close(fig)

def main():
    (ROOT / "outputs").mkdir(exist_ok=True)
    x, benchmark, diagnostics = load_returns()
    periods = {key: period_analysis(x, benchmark, key) for key in PERIODS}
    results = {"data": diagnostics, "periods": periods, "mardia": mardia(x),
               "factor_analysis": factor_analysis(x),
               "method": "Equal value weights of 10% per stock. PCA on daily log-return sample covariance. Historical VaR uses exact daily rebalanced simple returns.",
               "stress_rule": "Fixed calendar comparison: July-September 2024 versus October-December 2024. Candidate stress status assessed from observed benchmark return; not selected by maximising portfolio losses."}
    x.to_csv(ROOT / "outputs/log_returns.csv", float_format="%.12g")
    x.mean().rename("mean_daily_log_return").to_csv(ROOT / "outputs/mean_vector.csv")
    x.cov().to_csv(ROOT / "outputs/sample_covariance.csv")
    (ROOT / "outputs/results.json").write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
    (ROOT / "dist/results.js").write_text("window.RESULTS = " + json.dumps(results, allow_nan=False) + ";\n", encoding="utf-8")
    plots(results, x)
    print(json.dumps({"data": diagnostics, "periods": {k: {a: v[a] for a in ["n", "daily_volatility_pct", "historical_var95_pct", "mean_correlation", "benchmark_period_return_pct", "pc1_nifty_correlation"]} for k, v in periods.items()}, "mardia": results["mardia"], "fa_shared_pct": results["factor_analysis"]["shared_share_pct"]}, indent=2))

if __name__ == "__main__":
    main()
