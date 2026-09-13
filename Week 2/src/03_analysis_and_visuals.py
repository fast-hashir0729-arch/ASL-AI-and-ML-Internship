

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import (
    SALES_CLEAN_PATH,
    CUSTOMERS_CLEAN_PATH,
    VISUALS_DIR,
    print_section,
)

VISUALS_DIR.mkdir(parents=True, exist_ok=True)


def average_sales_by_region(df: pd.DataFrame) -> pd.Series:

    return df.groupby("region")["sales"].mean().sort_values(ascending=False)


def profit_margin_by_category(df: pd.DataFrame) -> pd.DataFrame:

    grouped = df.groupby("category").agg(
        total_sales=("sales", "sum"),
        total_profit=("profit", "sum"),
        order_count=("order_id", "count"),
    )
    grouped["profit_margin_pct"] = (
        grouped["total_profit"] / grouped["total_sales"] * 100
    ).round(2)
    return grouped.sort_values("profit_margin_pct", ascending=False)



def merge_sales_with_customers(
    sales_df: pd.DataFrame, customers_df: pd.DataFrame
) -> pd.DataFrame:

    return pd.merge(sales_df, customers_df, on="customer_id", how="left")



def sales_by_region_and_month(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["month"] = df["order_date"].dt.month_name()
    pivot = df.pivot_table(
        values="sales", index="region", columns="month", aggfunc="sum", fill_value=0
    )

    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    existing_months = [m for m in month_order if m in pivot.columns]
    return pivot[existing_months]



# Bonus: outlier detection (IQR method)

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> pd.DataFrame:

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"IQR bounds for '{column}': [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"Found {len(outliers)} outlier rows out of {len(df)}")
    return outliers


def plot_monthly_sales_trend(df: pd.DataFrame) -> None:

    monthly = df.set_index("order_date").resample("ME")["sales"].sum()
    plt.figure(figsize=(9, 5))
    plt.plot(monthly.index, monthly.values, marker="o")
    plt.title("Monthly Sales Trend (2023)")
    plt.xlabel("Month")
    plt.ylabel("Total Sales ($)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "01_monthly_sales_trend.png", dpi=150)
    plt.close()


def plot_sales_by_category(df: pd.DataFrame) -> None:

    category_totals = df.groupby("category")["sales"].sum().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(category_totals.index, category_totals.values, color="steelblue")
    plt.title("Total Sales by Product Category")
    plt.xlabel("Category")
    plt.ylabel("Total Sales ($)")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "02_sales_by_category.png", dpi=150)
    plt.close()


def plot_price_distribution(df: pd.DataFrame) -> None:

    plt.figure(figsize=(8, 5))
    plt.hist(df["price"], bins=20, color="darkorange", edgecolor="black")
    plt.title("Distribution of Unit Prices")
    plt.xlabel("Price ($)")
    plt.ylabel("Number of Orders")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "03_price_distribution.png", dpi=150)
    plt.close()


def plot_quantity_vs_sales(df: pd.DataFrame) -> None:

    plt.figure(figsize=(8, 5))
    plt.scatter(df["quantity"], df["sales"], alpha=0.6, color="seagreen")
    plt.title("Order Quantity vs. Total Sale Value")
    plt.xlabel("Quantity")
    plt.ylabel("Sales ($)")
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "04_quantity_vs_sales.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:

    numeric_cols = ["quantity", "price", "sales", "profit"]
    corr = df[numeric_cols].corr()

    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, label="Correlation coefficient")
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45)
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    plt.title("Correlation Heatmap — Numeric Columns")
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            plt.text(
                j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black"
            )
    plt.tight_layout()
    plt.savefig(VISUALS_DIR / "05_correlation_heatmap.png", dpi=150)
    plt.close()


def main() -> None:
    sales_df = pd.read_csv(SALES_CLEAN_PATH, parse_dates=["order_date"])
    customers_df = pd.read_csv(CUSTOMERS_CLEAN_PATH, parse_dates=["join_date"])

    print_section("GroupBy: average sales by region")
    print(average_sales_by_region(sales_df))

    print_section("GroupBy: profit margin by category")
    print(profit_margin_by_category(sales_df))

    print_section("Merge: sales joined with customers")
    merged_df = merge_sales_with_customers(sales_df, customers_df)
    print(f"Merged shape: {merged_df.shape}")
    print(merged_df[["order_id", "customer_id", "segment", "city"]].head())

    print_section("Pivot table: sales by region and month")
    pivot = sales_by_region_and_month(sales_df)
    print(pivot)

    print_section("Bonus: outlier detection on profit (IQR method)")
    outliers = detect_outliers_iqr(sales_df, "profit")
    print(outliers[["order_id", "category", "sales", "profit"]].head())

    print_section("Generating visualizations")
    plot_monthly_sales_trend(sales_df)
    plot_sales_by_category(sales_df)
    plot_price_distribution(sales_df)
    plot_quantity_vs_sales(sales_df)
    plot_correlation_heatmap(sales_df)  # bonus
    print(f"5 charts saved to {VISUALS_DIR}")


if __name__ == "__main__":
    main()
