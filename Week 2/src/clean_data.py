import pandas as pd

from utils import (
    SALES_RAW_PATH,
    CUSTOMERS_RAW_PATH,
    SALES_CLEAN_PATH,
    CUSTOMERS_CLEAN_PATH,
    PROCESSED_DATA_DIR,
    print_section,
)


def standardize_text_column(df: pd.DataFrame, column: str) -> pd.DataFrame:

    df[column] = df[column].astype(str).str.strip().str.title()
    return df


def parse_mixed_dates(df: pd.DataFrame, column: str) -> pd.DataFrame:

    df[column] = pd.to_datetime(df[column], format="mixed", errors="coerce")
    return df


def clean_price_column(df: pd.DataFrame, column: str) -> pd.DataFrame:

    df[column] = (
        df[column]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def fill_missing_by_group(df: pd.DataFrame, column: str, group_col: str) -> pd.DataFrame:

    df[column] = df.groupby(group_col)[column].transform(
        lambda s: s.fillna(s.median())
    )
    return df


def clean_sales_data(raw_df: pd.DataFrame) -> pd.DataFrame:

    df_clean = raw_df.copy() 

    df_clean = standardize_text_column(df_clean, "region")
    df_clean = standardize_text_column(df_clean, "category")
    df_clean = parse_mixed_dates(df_clean, "order_date")
    df_clean = clean_price_column(df_clean, "price")

    df_clean = fill_missing_by_group(df_clean, "price", "category")
    df_clean = fill_missing_by_group(df_clean, "profit", "category")

    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed = before_dedup - len(df_clean)
    print(f"Removed {removed} exact duplicate rows")


    df_clean["sales"] = (df_clean["price"] * df_clean["quantity"]).round(2)

    return df_clean


def clean_customers_data(raw_df: pd.DataFrame) -> pd.DataFrame:

    df_clean = raw_df.copy()
    df_clean = standardize_text_column(df_clean, "city")
    df_clean["join_date"] = pd.to_datetime(df_clean["join_date"], errors="coerce")
    df_clean = df_clean.drop_duplicates()
    return df_clean


def main() -> None:
    sales_raw = pd.read_csv(SALES_RAW_PATH)
    customers_raw = pd.read_csv(CUSTOMERS_RAW_PATH)

    print_section("Cleaning sales_data.csv")
    sales_clean = clean_sales_data(sales_raw)
    print(f"Shape before cleaning: {sales_raw.shape}")
    print(f"Shape after cleaning:  {sales_clean.shape}")
    print("\nMissing values after cleaning:")
    print(sales_clean.isnull().sum())
    print("\nUnique regions after standardizing:", sorted(sales_clean["region"].unique()))
    print("Unique categories after standardizing:", sorted(sales_clean["category"].unique()))

    print_section("Cleaning customers.csv")
    customers_clean = clean_customers_data(customers_raw)
    print(f"Shape before cleaning: {customers_raw.shape}")
    print(f"Shape after cleaning:  {customers_clean.shape}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sales_clean.to_csv(SALES_CLEAN_PATH, index=False)
    customers_clean.to_csv(CUSTOMERS_CLEAN_PATH, index=False)

    print_section("Cleaned files written")
    print(f"-> {SALES_CLEAN_PATH}")
    print(f"-> {CUSTOMERS_CLEAN_PATH}")


if __name__ == "__main__":
    main()
