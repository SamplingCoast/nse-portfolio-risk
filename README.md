# Multivariate Portfolio Risk Decomposition

Course project by Partha S G, Multivariate Techniques, CHRIST (Deemed to be University), Bengaluru.

The project asks how common movements contribute to an equal-weight portfolio of ten NSE stocks, and how the results differ in a fixed comparison window. It uses covariance PCA, exploratory factor analysis, Mardia diagnostics and historical Value at Risk. This is a retrospective academic analysis, not a prediction service.

## Data and reproduction

Yahoo Finance adjusted daily prices, 1 January 2023 to 31 December 2025. The fixed snapshot and its source metadata are in `data/`. Genuine extreme returns are retained. The included dashboard uses saved analysis outputs and needs no live data download.

## View the dashboard

Open `dist/index.html` in a browser, keeping `app.js` and `results.js` beside it. This works offline. For Codespaces or a local web server, run:

```text
python serve.py
```

Open port 8000. For assessment access in Codespaces, make that forwarded port Public, open it in a browser and test the URL in a private window. Codespaces must remain running for that URL to work.

## Reproduce the analysis

Python 3.12 was used. Viewing the saved dashboard needs no third-party packages. Recomputing uses:

```text
python -m pip install -r requirements.txt
python src/analysis.py
python -m unittest discover -s tests -v
```

Only run `python src/download_data.py` if intentionally obtaining a new snapshot. Yahoo's historical adjustments may change. The current report is tied to the checked-in CSV and the SHA256 in `data/provenance.json`.

## Main results

The fixed data contain 740 price dates and 739 return observations, without missing values. PC1 explains 37.28% of variance across the stocks and 97.13% of this equal-weight portfolio's variance. Daily log-return volatility is 0.7644%; historical one-day 95% VaR is 1.1059% using simple portfolio returns.

Compared with July-September 2024, October-December 2024 has a NIFTY decline of 8.39%, higher mean stock correlation and higher portfolio volatility. Historical 95% VaR is slightly lower (1.2050% versus 1.2873%). The mixed findings are retained rather than forcing all measures to agree.

Mardia's diagnostics reject joint normality; asymptotic p-values are qualified because financial time series may be dependent. The three-factor model is exploratory. Conditional covariance is interpreted as linear regression residual covariance when normality is not supported. Generalised variance describes the asset vector, not a weighted portfolio.

## Project organisation

```text
data/          Fixed adjusted prices and source provenance
src/           Download and mathematical analysis
outputs/       Results, mean vector, covariance, returns and labelled figures
dist/          Offline-capable interactive dashboard
tests/         Independent numerical checks
.devcontainer/ Codespaces configuration for future launches
serve.py       Dashboard web server
requirements.txt  Tested direct dependency versions
```

The six checks cover SVD versus eigenvalues, observed portfolio variance versus risk contributions, regression versus residual-covariance formula, unit invariance of Mardia statistics, singular input rejection and finite data. All six passed during preparation. The dashboard's three periods and expanded diagnostics were checked in a desktop and mobile browser layout without JavaScript errors or horizontal page overflow.

## Submission evidence

- Public repository URL: https://github.com/SamplingCoast/nse-portfolio-risk
- Public running dashboard URL: https://scaling-space-garbanzo-7v5jqx5v4gv7cp666-8000.app.github.dev/
- GitHub Student Developer Pack: Pending
- Actual Pack benefit used: pending student confirmation. Proposed use is Codespaces under the verified student account to run this project.
- Evidence: save Student_Pack_Approved.png, Codespaces_Run.png and Dashboard_Live.png outside this public repositor.

The supplied Git history records real implementation stages authored by the project assistant. It must not be represented as a fabricated student-only development history. Student review, understanding and account actions remain necessary.

## References

- Yahoo Finance historical prices: individual links, dates and file checksum in `data/provenance.json`.
- [yfinance download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
- [NIST Principal Components](https://www.itl.nist.gov/div898/software/dataplot/refman2/ch4/prin_com.pdf).
- [MRC CBU Mardia diagnostics](https://imaging.mrc-cbu.cam.ac.uk/statswiki/FAQ/Rmardia).
- [scikit-learn FactorAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html).
- [GitHub student application](https://docs.github.com/en/education/about-github-education/github-education-for-students/apply-to-github-education-as-a-student).
- [GitHub port forwarding](https://docs.github.com/en/codespaces/developing-in-a-codespace/forwarding-ports-in-your-codespace).
