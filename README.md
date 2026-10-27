# MotorPH Business Analysis: MO-IT106 Terminal Assessment

Analysis of MotorPH's product catalogue and Q3 2025 sales data (June to August 2025) in Python, with a written Business Analysis Report.

- **Author:** Kayel Niccolo C. Astrera, BSIT (Network and Cybersecurity), Section H2101
- **Course:** MO-IT106 Data Analytics Fundamentals, Mapua-Malayan Digital College
- **Term:** Term 1, Academic Year 2026-2027

## What this project answers

1. What does MotorPH sell, and at what prices?
2. How are those products selling (by month, category, product, client type and payment method)?
3. Did sales change between July and August 2025, and does price explain how many units sell?

## Repository contents

| File | Purpose |
| --- | --- |
| `MO-IT106 Terminal Assessment: MotorPH Business Analysis Report - H2101 - Astrera, Kayel Niccolo C..docx` | The full report: Executive Summary, Introduction, Methodology, Data Analysis and Interpretation, Conclusions and Business Insights, Appendices |
| `MotorPH_Final_Dashboard.py` | Milestone 2 dashboard: loads and validates the data, builds summary tables and charts, exports the dashboard and an Excel workbook |
| `MotorPH_Inferential_Analysis.py` | Inferential analysis: cleans the raw Q3 sales file, draws normality plots, runs the hypothesis tests |
| `MotorPH_Products_Preprocessed.csv` | Product catalogue: 50 products, 6 columns |
| `MotorPH_Sales_Data-3rd_Quarter-Year_2025_1.csv` | Raw Q3 2025 sales: 1,000 rows, 7 columns |
| `MotorPH_Sales_Preprocessed.csv` | Cleaned sales: 994 rows, 7 columns |

## Data

| Dataset | Rows | Columns |
| --- | --- | --- |
| Products (preprocessed) | 50 | Product ID Number, Product Name, Product Type, Unit Price, Date of Manufacturing, Date of Acquisition |
| Sales (raw Q3) | 1,000 | date, client_type, product, unitprice, quantity, total, payment |
| Sales (preprocessed) | 994 | Date, Client Type, Product Name, Unit Price, Quantity, Total, Payment Method |

Sales cover 1 June to 31 August 2025, plus one order dated 13 January 2025. There are no September records. The files contain no stock quantities and no costs, so the analysis describes sales and the catalogue, not stock levels or profit.

### How the raw file becomes the clean file

| Problem in the raw file | Rows | Fix |
| --- | --- | --- |
| Dates in other layouts (`07-25-25`, `Aug-10-2025`, `13/01/2025`) | 4 | Converted to one date format |
| Missing or impossible dates (blank, `2025/07/32`, `2025-08-00`, `08-40-2025`) | 6 | Removed, leaving 994 rows |
| Product names with a damaged last character (for example `Benelli 502x`) | 15 | Matched to the catalogue name |
| Unit prices that disagree with the catalogue | 20 | Set to the catalogue price; Total = Unit Price x Quantity |
| Blank client type or payment method | 10 each | Labelled "Unknown" and kept |
| Inconsistent capitalization (`Credit card`) | all | Standardized |

## Requirements

- Python 3.9 or newer
- `pandas`, `numpy`, `matplotlib`
- `openpyxl`, only for the Excel export in the dashboard script (skip it with `--no-excel`)

Install with `pip install pandas numpy matplotlib openpyxl`. The inferential script needs no other packages: its statistical tests are written in plain Python and were checked against SciPy.

## How to run

Put the scripts and the CSV files in the same folder, then:

```bash
# Milestone 2 dashboard (reads MotorPH_Products_Preprocessed.csv and MotorPH_Sales_Preprocessed.csv)
python MotorPH_Final_Dashboard.py

# Inferential analysis (reads the raw Q3 sales file and cleans it)
python MotorPH_Inferential_Analysis.py
```

Useful options:

| Option | Dashboard | Inferential |
| --- | --- | --- |
| `--data-dir PATH` | folder with the CSV files | folder with the CSV files |
| `--output-dir PATH` | where results are saved (default `dashboard_output`) | where results are saved (default `inferential_output`) |
| `--no-show` | save files without opening chart windows | do not open the plot window |
| `--no-excel` | skip the Excel workbook | n/a |
| `--sales-file NAME` | n/a | use another sales CSV, for example `MotorPH_Sales_Preprocessed.csv` |

If the raw Q3 file is not in the folder, the inferential script falls back to `MotorPH_Sales_Preprocessed.csv`. Both give the same results.

## Outputs

**Dashboard script** (`dashboard_output/`): `MotorPH_Dashboard.png` and `.pdf`, `MotorPH_Dashboard_Inventory.png`, `product_distribution_by_category.png`, `MotorPH_Dashboard.xlsx`, and summary tables in `tables/`.

**Inferential script** (`inferential_output/`): `normality_plots.png`, `normality_tests.csv`, `month_comparison_tests.csv`, `conclusions.txt`.

Both scripts print a data-quality report in the terminal before the results.

## Method in brief

1. **Load and validate:** stop with a clear message if a file or column is missing.
2. **Clean:** fix types, dates, names, prices and blanks; check that every total equals price x quantity and every sale matches a catalogue product.
3. **Describe:** product counts and prices; sales by month, week, client type, payment method, product and category.
4. **Test normality:** histograms with a fitted normal curve, Q-Q plots and the D'Agostino-Pearson test (alpha = 0.05).
5. **Compare July and August:** Mann-Whitney U (main test, because the data are not normal), with Welch t-test and Levene's test as cross-checks.
6. **Check price against demand:** Spearman rank correlation between unit price and units sold across the 50 products.

## Key results

| Result | Value |
| --- | --- |
| Products / categories | 50 / 17 |
| Mean unit price | ₱257,872 (range ₱48,000 to ₱995,000) |
| Sales orders / revenue / units | 994 / ₱4.46 billion / 17,619 |
| Monthly revenue (June, July, August) | ₱1.52B, ₱1.49B, ₱1.45B |
| Orders (June, July, August) | 342, 331, 320 |
| Normality (all four samples) | Rejected, p < 0.001 |
| July vs August order value (Mann-Whitney U) | p = 0.654, no significant difference |
| July vs August units per order (Mann-Whitney U) | p = 0.738, no significant difference |
| Unit price vs units sold (Spearman) | rho = -0.073, p = 0.613, no relationship |
| Revenue per model | Heritage ₱276.6M, Scooter ₱62.3M |

The August dip comes from fewer orders, not smaller ones. A non-significant test means no evidence of a difference, not proof the months are equal.

## Limitations

- Three months of sales and no September data, so no seasonal or long-term trend can be tested.
- No stock quantities, costs or margins: the data cannot show over- or understock or profit.
- The "total inventory cost" (₱12,893,600) is the sum of one unit price per model, not the value of stock on hand.
- The tests compare individual orders; they do not test whether the number of orders changed.
- The September projection (about ₱1.41B to ₱1.45B) is a simple three-month extrapolation, not a forecast.

## Where to find things in the report

| Report section | Contents |
| --- | --- |
| I. Executive Summary | Background, scope, key findings, recommendations |
| II. Introduction | Purpose, data overview, decision support |
| III. Methodology | Scope, data preparation, analysis process, tools |
| IV. Data Analysis and Interpretation | A. Descriptive, B. Sales, C. Inferential (Tests 1 to 4) |
| V. Conclusions and Business Insights | A. Findings, B. Predictions and recommendations |
| VI. Appendices | Dataset links, code, supporting outputs |
