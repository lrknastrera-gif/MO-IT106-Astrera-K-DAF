"""
MotorPH Inferential Analysis  -  MO-IT106 Terminal Assessment
=============================================================

Question: did MotorPH sales change between July and August 2025?

What this script does
---------------------
1. LOAD       Reads the two preprocessed CSV files and stops with a clear message
              if a file or column is missing.
2. PREPARE    Cleans the sales data (types, blanks, duplicates, Total check) and
              attaches each sale's Product Type.
3. DESCRIBE   Prints monthly summaries (June, July, August).
4. NORMALITY  Histograms with a fitted normal curve + Q-Q plots, and the
              D'Agostino-Pearson test, for July and August (Sales value in PHP
              and Units per order).
5. HYPOTHESIS Mann-Whitney U (main test, no normality assumption), Welch t-test and
              Levene test (cross-checks) for July vs August; Spearman correlation
              between unit price and units sold across products.
6. EXPORT     Saves the plot (PNG), the results table (CSV) and a text report.

Run it
------
    python MotorPH_Inferential_Analysis.py
    python MotorPH_Inferential_Analysis.py --data-dir path/to/csvs --output-dir results
    python MotorPH_Inferential_Analysis.py --no-show

The two CSV files must sit in the data folder (default: the folder of this script):
    MotorPH_Products_Preprocessed.csv
    MotorPH_Sales_Preprocessed.csv

Requirements: pandas, numpy, matplotlib (the same packages as the dashboard script).
No SciPy needed: every statistical test is written out below using only the Python
standard library and numpy, and was checked against SciPy's results.
"""

from __future__ import annotations

import argparse
import math
import sys
from statistics import NormalDist
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALPHA = 0.05  # significance level used for every test

PRODUCT_COLS = ["Product ID Number", "Product Name", "Product Type", "Unit Price",
                "Date of Manufacturing", "Date of Acquisition"]
SALES_COLS = ["Date", "Client Type", "Product Name", "Unit Price", "Quantity",
              "Total", "Payment Method"]

# The two months being compared, and the two variables tested in each.
MONTHS = {"July": 7, "August": 8}
VARIABLES = {"Sales value (PHP)": "Total", "Units": "Quantity"}


class DataError(Exception):
    """Raised when an input file cannot be used (missing, empty, wrong columns)."""



# ----------------------------------------------------------------------------
# 0. STATISTICS HELPERS (replace SciPy: standard library + numpy only)
# ----------------------------------------------------------------------------
def norm_sf(z: float) -> float:
    """Upper-tail probability of the standard normal distribution."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value of Student's t distribution."""
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def f_sf(f: float, d1: float, d2: float) -> float:
    """Upper-tail probability of the F distribution."""
    return betainc(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * f))


def dagostino_pearson(x) -> tuple[float, float]:
    """D'Agostino-Pearson omnibus test of normality: returns (K2, p-value).

    Combines a skewness test and a kurtosis test. H0: the data are normally distributed.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    d = x - x.mean()
    m2, m3, m4 = (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()
    skew, kurt = m3 / m2 ** 1.5, m4 / m2 ** 2

    y = skew * math.sqrt((n + 1) * (n + 3) / (6.0 * (n - 2)))
    beta2 = 3.0 * (n ** 2 + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2.0) * (n + 5) * (n + 7) * (n + 9))
    w2 = -1 + math.sqrt(2 * (beta2 - 1))
    delta = 1 / math.sqrt(0.5 * math.log(w2))
    alpha = math.sqrt(2.0 / (w2 - 1))
    y = y if y != 0 else 1.0
    z_skew = delta * math.log(y / alpha + math.sqrt((y / alpha) ** 2 + 1))

    e = 3.0 * (n - 1) / (n + 1)
    var_b2 = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) ** 2 * (n + 3) * (n + 5))
    xx = (kurt - e) / math.sqrt(var_b2)
    sqrt_beta1 = (6.0 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9))
                  * math.sqrt(6.0 * (n + 3) * (n + 5) / (n * (n - 2) * (n - 3))))
    a = 6.0 + 8.0 / sqrt_beta1 * (2.0 / sqrt_beta1 + math.sqrt(1 + 4.0 / sqrt_beta1 ** 2))
    term1 = 1 - 2 / (9.0 * a)
    denom = 1 + xx * math.sqrt(2 / (a - 4.0))
    term2 = math.copysign(((1 - 2.0 / a) / abs(denom)) ** (1 / 3), denom)
    z_kurt = (term1 - term2) / math.sqrt(2 / (9.0 * a))

    k2 = z_skew ** 2 + z_kurt ** 2
    return k2, math.exp(-k2 / 2.0)          # chi-square with 2 degrees of freedom


def skewness(x) -> float:
    """Sample skewness (0 = symmetric, positive = long right tail)."""
    d = np.asarray(x, dtype=float) - np.mean(x)
    return float((d ** 3).mean() / (d ** 2).mean() ** 1.5)


def mann_whitney_u(a, b) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test (normal approximation, tie and continuity corrected).

    H0: both samples come from the same distribution. Returns (U for sample a, p-value).
    """
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    n1, n2 = len(a), len(b)
    n = n1 + n2
    ranks = pd.concat([a, b], ignore_index=True).rank(method="average")
    u1 = float(ranks.iloc[:n1].sum() - n1 * (n1 + 1) / 2.0)
    u = max(u1, n1 * n2 - u1)
    ties = pd.concat([a, b]).value_counts().to_numpy(dtype=float)
    tie_term = float((ties ** 3 - ties).sum())
    sd = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1))))
    z = (u - n1 * n2 / 2.0 - 0.5) / sd
    return u1, min(1.0, 2.0 * norm_sf(z))


def welch_t(a, b) -> tuple[float, float]:
    """Welch's t-test (unequal variances): returns (t, two-sided p-value)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return t, t_two_sided_p(t, df)


def levene(a, b) -> tuple[float, float]:
    """Levene / Brown-Forsythe test that two groups have equal spread: (F, p-value)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    za, zb = np.abs(a - np.median(a)), np.abs(b - np.median(b))
    n, grand = len(za) + len(zb), np.concatenate([za, zb]).mean()
    between = len(za) * (za.mean() - grand) ** 2 + len(zb) * (zb.mean() - grand) ** 2
    within = ((za - za.mean()) ** 2).sum() + ((zb - zb.mean()) ** 2).sum()
    f = (between / 1.0) / (within / (n - 2))
    return f, f_sf(f, 1, n - 2)


def spearman(x, y) -> tuple[float, float]:
    """Spearman rank correlation and its two-sided p-value."""
    rx, ry = pd.Series(x, dtype=float).rank(), pd.Series(y, dtype=float).rank()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    t = rho * math.sqrt((len(rx) - 2) / (1 - rho ** 2))
    return rho, t_two_sided_p(t, len(rx) - 2)


# ----------------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------------
def load_csv(path: Path, required: list[str]) -> pd.DataFrame:
    """Read a CSV file and confirm it exists, is not empty and has the required columns."""
    if not path.exists():
        found = sorted(p.name for p in path.parent.glob("*.csv")) if path.parent.exists() else []
        raise DataError(f"File not found: {path}\n"
                        f"Expected file name: {path.name}\n"
                        f"CSV files found in {path.parent}: {found if found else 'none'}\n"
                        f"Rename the file or pass --data-dir.")
    data = pd.read_csv(path)
    if data.empty:
        raise DataError(f"{path.name} is empty.")
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise DataError(f"{path.name} is missing columns {missing}. Found: {list(data.columns)}")
    return data


# ----------------------------------------------------------------------------
# 2. PREPARE
# ----------------------------------------------------------------------------
def prepare(products_raw: pd.DataFrame, sales_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Clean both datasets and return (products, sales, data-quality report)."""
    report: dict[str, int] = {}

    products = products_raw[PRODUCT_COLS].copy()
    products["Unit Price"] = pd.to_numeric(products["Unit Price"], errors="coerce")
    report["Product rows loaded"] = len(products)
    products = products.dropna(subset=["Product Name", "Product Type", "Unit Price"])
    products = products.drop_duplicates(subset="Product ID Number")
    report["Product rows used"] = len(products)

    sales = sales_raw[SALES_COLS].copy()
    report["Sales rows loaded"] = len(sales)
    for col in ["Client Type", "Product Name", "Payment Method"]:
        sales[col] = sales[col].astype("string").str.strip().replace("", pd.NA)
    sales["Client Type"] = sales["Client Type"].str.title()
    sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")
    for col in ["Unit Price", "Quantity", "Total"]:
        sales[col] = pd.to_numeric(sales[col], errors="coerce")

    unusable = sales[["Date", "Product Name", "Unit Price", "Quantity"]].isna().any(axis=1)
    unusable |= (sales["Quantity"] <= 0) | (sales["Unit Price"] <= 0)
    report["Rows removed (missing/invalid date, product, price or quantity)"] = int(unusable.sum())
    sales = sales[~unusable].copy()

    for col in ["Client Type", "Payment Method"]:
        sales[col] = sales[col].fillna("Unknown")

    expected = sales["Unit Price"] * sales["Quantity"]
    report["Totals that did not equal Unit Price x Quantity"] = int((sales["Total"] != expected).sum())
    sales["Total"] = expected

    report["Duplicate sales rows removed"] = int(sales.duplicated().sum())
    sales = sales.drop_duplicates()

    lookup = products.set_index("Product Name")["Product Type"]
    sales["Product Type"] = sales["Product Name"].map(lookup).fillna("Uncategorized")
    report["Sales not matched to a catalogue product"] = int((sales["Product Type"] == "Uncategorized").sum())
    report["Sales rows used"] = len(sales)
    return products, sales.sort_values("Date").reset_index(drop=True), report


# ----------------------------------------------------------------------------
# 3. DESCRIBE
# ----------------------------------------------------------------------------
def monthly_summary(sales: pd.DataFrame) -> pd.DataFrame:
    """Orders, units, revenue and order-size statistics for every month in the data."""
    grouped = sales.groupby(sales["Date"].dt.to_period("M"))
    table = grouped.agg(Orders=("Total", "size"), Units=("Quantity", "sum"), Revenue=("Total", "sum"),
                        Mean_Order=("Total", "mean"), Median_Order=("Total", "median"),
                        Std_Order=("Total", "std"), Mean_Units=("Quantity", "mean"))
    return table.round(2)


def month_samples(sales: pd.DataFrame, column: str) -> dict[str, pd.Series]:
    """Values of `column` for each month being compared (2025 only)."""
    in_2025 = sales[sales["Date"].dt.year == 2025]
    return {name: in_2025.loc[in_2025["Date"].dt.month == num, column] for name, num in MONTHS.items()}


# ----------------------------------------------------------------------------
# 4. NORMALITY
# ----------------------------------------------------------------------------
def plot_normality(sales: pd.DataFrame, path: Path) -> plt.Figure:
    """2 x 4 figure: histogram + normal fit and Q-Q plot, for each month and variable."""
    fig, axes = plt.subplots(len(MONTHS), 4, figsize=(20, 8.5))
    for r, month in enumerate(MONTHS):
        for k, (label, column) in enumerate(VARIABLES.items()):
            x = month_samples(sales, column)[month].to_numpy(dtype=float)
            ax_hist, ax_qq = axes[r, 2 * k], axes[r, 2 * k + 1]

            ax_hist.hist(x, bins=20, density=True, color="#7094B8", edgecolor="white")
            grid = np.linspace(x.min(), x.max(), 300)
            ax_hist.plot(grid, np.exp(-0.5 * ((grid - x.mean()) / x.std(ddof=1)) ** 2) / (x.std(ddof=1) * math.sqrt(2 * math.pi)), color="red", label="Normal fit")
            ax_hist.set_title(f"{month}: {label}")
            ax_hist.legend()

            ordered = np.sort(x)
            theory = np.array([NormalDist().inv_cdf((i - 0.5) / len(x)) for i in range(1, len(x) + 1)])
            slope, intercept = np.polyfit(theory, ordered, 1)
            ax_qq.plot(theory, ordered, "o", color="blue", markersize=5)
            ax_qq.plot(theory, slope * theory + intercept, color="red")
            ax_qq.set_title(f"{month}: Q-Q plot ({label})")
            ax_qq.set_xlabel("Theoretical quantiles")
            ax_qq.set_ylabel("Ordered Values")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    return fig


def normality_tests(sales: pd.DataFrame) -> pd.DataFrame:
    """D'Agostino-Pearson test of normality for each month and variable.

    H0: the variable is normally distributed.   H1: it is not.
    """
    rows = []
    for label, column in VARIABLES.items():
        for month, x in month_samples(sales, column).items():
            k2, p = dagostino_pearson(x)
            rows.append({"Test": "Normality", "Variable": label, "Month": month, "n": len(x),
                         "Skewness": round(skewness(x), 3),
                         "D'Agostino-Pearson K2": round(k2, 2), "p-value": p,
                         "Decision": "Reject H0 (not normal)" if p < ALPHA else "Fail to reject H0"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 5. HYPOTHESIS TESTS
# ----------------------------------------------------------------------------
def compare_months(sales: pd.DataFrame, normal: pd.DataFrame) -> pd.DataFrame:
    """July vs August for each variable.

    H0: July and August values come from the same distribution (no difference).
    H1: they come from different distributions.
    Mann-Whitney U is the main test because the normality tests reject normality.
    Welch t-test and Levene's test are shown as cross-checks.
    """
    rows = []
    for label, column in VARIABLES.items():
        jul, aug = (month_samples(sales, column)[m].astype(float) for m in MONTHS)
        u, p_u = mann_whitney_u(jul, aug)
        t, p_t = welch_t(jul, aug)
        lev, p_l = levene(jul, aug)
        all_normal = bool((normal.loc[normal["Variable"] == label, "p-value"] >= ALPHA).all())
        rows.append({"Test": "Mann-Whitney U (main)", "Variable": label,
                     "July median": float(jul.median()), "August median": float(aug.median()),
                     "Statistic": float(u), "p-value": float(p_u),
                     "Decision": "Reject H0" if p_u < ALPHA else "Fail to reject H0"})
        rows.append({"Test": "Welch t-test (cross-check)", "Variable": label,
                     "July median": float(jul.mean()), "August median": float(aug.mean()),
                     "Statistic": float(t), "p-value": float(p_t),
                     "Decision": "Reject H0" if p_t < ALPHA else "Fail to reject H0"})
        rows.append({"Test": "Levene equal-variance (cross-check)", "Variable": label,
                     "July median": float(jul.std()), "August median": float(aug.std()),
                     "Statistic": float(lev), "p-value": float(p_l),
                     "Decision": "Reject H0" if p_l < ALPHA else "Fail to reject H0"})
        if all_normal:
            print(f"Note: {label} passed the normality test; the Welch t-test is also appropriate.")
    out = pd.DataFrame(rows)
    out = out.rename(columns={"July median": "July (median / mean / SD)", "August median": "August (median / mean / SD)"})
    return out


def price_vs_units(products: pd.DataFrame, sales: pd.DataFrame) -> dict:
    """Spearman correlation between a product's unit price and the units sold.

    H0: no monotonic relationship between price and units sold (rho = 0).
    H1: there is a relationship (rho != 0).
    """
    units = sales.groupby("Product Name")["Quantity"].sum()
    table = products.set_index("Product Name")[["Unit Price"]].join(units).fillna({"Quantity": 0})
    rho, p = spearman(table["Unit Price"], table["Quantity"])
    return {"n products": len(table), "rho": float(rho), "p-value": float(p),
            "Decision": "Reject H0" if p < ALPHA else "Fail to reject H0"}


# ----------------------------------------------------------------------------
# 6. REPORT
# ----------------------------------------------------------------------------
def interpret(comparison: pd.DataFrame, corr: dict) -> list[str]:
    """Plain-language conclusions generated from the test results."""
    lines = []
    main = comparison[comparison["Test"].str.startswith("Mann-Whitney")]
    for _, r in main.iterrows():
        if r["p-value"] < ALPHA:
            lines.append(f"{r['Variable']}: July and August differ significantly (p = {r['p-value']:.3f}).")
        else:
            lines.append(f"{r['Variable']}: no significant difference between July and August "
                         f"(p = {r['p-value']:.3f}). This is 'no evidence of a difference', not proof they are equal.")
    direction = "no" if corr["p-value"] >= ALPHA else "a"
    lines.append(f"Price vs units sold across {corr['n products']} products: rho = {corr['rho']:.2f}, "
                 f"p = {corr['p-value']:.2f} -> {direction} significant relationship.")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="MotorPH inferential analysis (July vs August 2025).")
    here = Path(__file__).resolve().parent
    parser.add_argument("--data-dir", type=Path, default=here, help="folder with the two CSV files")
    parser.add_argument("--output-dir", type=Path, default=here / "inferential_output", help="where results are saved")
    parser.add_argument("--no-show", action="store_true", help="do not open the plot window")
    args = parser.parse_args()

    try:
        products_raw = load_csv(args.data_dir / "MotorPH_Products_Preprocessed.csv", PRODUCT_COLS)
        sales_raw = load_csv(args.data_dir / "MotorPH_Sales_Preprocessed.csv", SALES_COLS)
    except DataError as err:
        print(f"ERROR: {err}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200, "display.max_columns", 20)

    products, sales, quality = prepare(products_raw, sales_raw)
    print("--- Data-quality report ---")
    for key, value in quality.items():
        print(f"{key}: {value:,}")

    print("\n--- Monthly summary ---")
    print(monthly_summary(sales).to_string())

    fig = plot_normality(sales, args.output_dir / "normality_plots.png")
    print(f"\nSaved {args.output_dir / 'normality_plots.png'}")

    normal = normality_tests(sales)
    print("\n--- Test 1: Normality (H0: normally distributed) ---")
    print(normal.to_string(index=False))

    comparison = compare_months(sales, normal)
    print("\n--- Tests 2-3: July vs August (H0: same distribution) ---")
    print(comparison.to_string(index=False))

    corr = price_vs_units(products, sales)
    print("\n--- Test 4: Spearman, unit price vs units sold ---")
    print(corr)

    conclusions = interpret(comparison, corr)
    print("\n--- Conclusions (alpha = 0.05) ---")
    for line in conclusions:
        print("-", line)

    normal.to_csv(args.output_dir / "normality_tests.csv", index=False)
    comparison.to_csv(args.output_dir / "month_comparison_tests.csv", index=False)
    (args.output_dir / "conclusions.txt").write_text("\n".join(conclusions), encoding="utf-8")
    print(f"\nSaved results in {args.output_dir}")

    if not args.no_show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
