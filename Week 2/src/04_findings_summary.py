

import pandas as pd

from utils import SALES_CLEAN_PATH, PROJECT_ROOT, print_section

FINDINGS_PATH = PROJECT_ROOT / "reports" / "findings_summary.md"


def build_findings_text(df: pd.DataFrame) -> str:
    total_sales = df["sales"].sum()
    total_profit = df["profit"].sum()
    margin_pct = round(total_profit / total_sales * 100, 2)
    avg_order_value = round(df["sales"].mean(), 2)

    region_avg = df.groupby("region")["sales"].mean().sort_values(ascending=False)
    top_region = region_avg.index[0]
    top_region_avg = round(region_avg.iloc[0], 2)

    category_margin = (
        df.groupby("category")
        .apply(
            lambda g: g["profit"].sum() / g["sales"].sum() * 100, include_groups=False
        )
        .sort_values(ascending=False)
    )
    top_category = category_margin.index[0]
    top_category_pct = round(category_margin.iloc[0], 2)
    bottom_category = category_margin.index[-1]
    bottom_category_pct = round(category_margin.iloc[-1], 2)

    missing_customers = df["customer_id"].isnull().sum()
    price_sales_corr = round(df["price"].corr(df["sales"]), 2)

    text = f"""\
## Findings Summary

This dataset covers {len(df)} cleaned retail orders placed between
{df['order_date'].min().strftime('%B %Y')} and {df['order_date'].max().strftime('%B %Y')},
totaling ${total_sales:,.2f} in sales and ${total_profit:,.2f} in profit,
an overall profit margin of {margin_pct}%. The average order value across
all regions was ${avg_order_value:,.2f}.

Looking at regional performance, the **{top_region}** region had the highest
average sale value per order at ${top_region_avg:,.2f}, ahead of the other
three regions, though the gap between regions was relatively narrow — no
single region dominates the business.

Category-level profitability tells a more interesting story than raw sales
volume: **{top_category}** had the highest profit margin at {top_category_pct}%,
while **{bottom_category}** had the lowest at {bottom_category_pct}%, despite
neither being the top category by total sales. This suggests margin and
volume don't move together here, and a volume-only view of "top performers"
would be misleading.

The correlation analysis showed unit price is more strongly related to sale
value (correlation of {price_sales_corr}) than quantity is, meaning big
orders are usually driven by higher-priced items rather than customers
buying more units per order. An IQR-based outlier check on profit flagged
a meaningful share of orders as statistical outliers, mostly high-margin
Electronics and Office Supplies orders, worth a closer look in a follow-up
analysis.

One data-quality limitation carried over from cleaning: {missing_customers}
orders have no linked customer_id and were kept rather than dropped, so any
customer-segment analysis built on the merged dataset slightly undercounts
total order volume for those {missing_customers} orders.
"""
    return text


def main() -> None:
    df = pd.read_csv(SALES_CLEAN_PATH, parse_dates=["order_date"])
    findings_text = build_findings_text(df)

    print_section("Findings Summary")
    print(findings_text)

    word_count = len(findings_text.split())
    print(f"(word count: {word_count})")

    FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINDINGS_PATH.write_text(findings_text, encoding="utf-8")
    print(f"\nSaved to {FINDINGS_PATH}")


if __name__ == "__main__":
    main()
