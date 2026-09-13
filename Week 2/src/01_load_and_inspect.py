
import pandas as pd

from utils import SALES_RAW_PATH, CUSTOMERS_RAW_PATH, print_section


def inspect_dataframe(df: pd.DataFrame, name: str) -> None:
    """Print a standard profiling report for any DataFrame."""
    print_section(f"{name} — shape: {df.shape}")

    print("\n-- dtypes --")
    print(df.dtypes)

    print("\n-- first 5 rows --")
    print(df.head())

    print("\n-- missing values per column --")
    print(df.isnull().sum())

    print("\n-- duplicate rows --")
    print(f"{df.duplicated().sum()} exact duplicate rows found")


def main() -> None:
    sales_df = pd.read_csv(SALES_RAW_PATH)
    customers_df = pd.read_csv(CUSTOMERS_RAW_PATH)

    inspect_dataframe(sales_df, "sales_data.csv (raw)")
    inspect_dataframe(customers_df, "customers.csv (raw)")

    # Observations worth acting on in the cleaning phase:
    #   - `order_date` is stored as text in two different formats
    #     (YYYY-MM-DD and MM/DD/YYYY) -> needs a single parsed datetime dtype.
    #   - `price` is stored as text for some rows, prefixed with "$" and
    #     containing thousands separators -> needs to become float.
    #   - `region` and `category` contain inconsistent casing and stray
    #     whitespace (" north ", "NORTH", "north") -> needs standardizing.
    #   - `profit` and `price` both contain missing values -> needs an
    #     explicit fill/drop strategy, not silent removal.
    #   - `customer_id` has a handful of missing values -> orders that
    #     can't be linked to a customer for the merge step.
    #   - There are exact duplicate rows -> need to be dropped once,
    #     with the count logged so it's auditable.
    print_section("Inspection complete — see comments in this file for findings")


if __name__ == "__main__":
    main()
