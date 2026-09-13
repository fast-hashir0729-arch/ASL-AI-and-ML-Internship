# AI/ML Internship — Week 2: Data Wrangling & EDA with NumPy and Pandas

**Advance Soft Logics — Remote Internship Program (AI/ML Track)**

## Project Objective

This project performs a complete Exploratory Data Analysis (EDA) on a retail
sales dataset: loading raw data, profiling it, cleaning it, analyzing it with
groupby/merge/pivot operations, visualizing it, and summarizing the findings
in plain English — the same workflow a junior data analyst would follow on
a real dataset.

## Dataset

- **`data/raw/sales_data.csv`** — 332 retail orders (10 columns: order id,
  date, region, category, product, customer id, quantity, price, sales,
  profit). Deliberately messy: duplicate rows, missing values, mixed date
  formats, currency-string prices, and inconsistent text casing.
- **`data/raw/customers.csv`** — 150 customers (customer id, name, segment,
  city, join date), used for the merge step.

Both datasets were generated to closely mirror the structure and typical
data-quality issues of a real exported retail sales report, since a
Kaggle/UCI download was not available in this environment. They satisfy the
assignment's validation rules (≥200 rows, ≥5 columns).

## Project Structure

```
├── data/
│   ├── raw/                  # untouched original data
│   └── processed/            # cleaned output from the pipeline
├── src/
│   ├── utils.py               # shared paths, no hard-coded absolute paths
│   ├── 01_load_and_inspect.py # Phase 2: load & profile raw data
│   ├── 02_clean_data.py       # Phase 3: cleaning pipeline
│   ├── 03_analysis_and_visuals.py  # Phase 4: groupby/merge/pivot + charts
│   └── 04_findings_summary.py # Phase 5: plain-English findings
├── visuals/                   # exported chart PNGs
├── reports/                   # findings_summary.md + final Word report
├── requirements.txt
└── README.md
```

## Setup & Run Instructions

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run each phase in order
cd src
python 01_load_and_inspect.py
python 02_clean_data.py
python 03_analysis_and_visuals.py
python 04_findings_summary.py
```

## Implemented Features

- Loading and profiling raw CSV data (shape, dtypes, missing values, duplicates)
- Documented cleaning pipeline: text standardization, mixed-date parsing,
  currency-string conversion, group-wise median imputation, duplicate removal
- GroupBy aggregation: average sales by region; profit margin by category
- Merge: sales joined with customers (left join, preserves unmatched sales)
- Pivot table: total sales by region and month
- 5 labeled visualizations: line, bar, histogram, scatter, and heatmap
- Plain-English findings summary referencing real computed numbers

## Optional / Bonus Features Implemented

- **Outlier detection (IQR method)** on the `profit` column
- **Correlation heatmap** between all numeric columns
- **Exported cleaned dataset** to `data/processed/`
- **Reusable functions** for every repeated cleaning/analysis step

## Top Findings

1. The **South** region had the highest average order value (~$1,678), but
   the gap across regions was narrow — no single region dominates.
2. **Office Supplies** had the best profit margin (14.86%) despite not being
   the top category by raw sales — margin and volume don't move together.
3. Unit price correlates with total sale value (r = 0.74) much more strongly
   than quantity does, meaning larger orders come from pricier items, not
   bulk buying.

See `reports/findings_summary.md` for the full 150–300 word write-up and
`reports/Week2_EDA_Report.docx` for the formatted final report.

## Known Limitations

- 8 orders in the cleaned dataset have no linked `customer_id` (kept rather
  than dropped to avoid losing valid sales records); any customer-segment
  breakdown built from the merged data slightly undercounts these orders.
- The dataset is synthetically generated rather than sourced externally,
  since the environment used to complete this assignment has no general
  internet access to Kaggle/UCI. It was constructed to realistically mirror
  the structure and messiness of an exported retail sales report.
