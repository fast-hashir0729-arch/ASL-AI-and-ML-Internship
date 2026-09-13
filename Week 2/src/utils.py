
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
VISUALS_DIR = PROJECT_ROOT / "visuals"

SALES_RAW_PATH = RAW_DATA_DIR / "sales_data.csv"
CUSTOMERS_RAW_PATH = RAW_DATA_DIR / "customers.csv"
SALES_CLEAN_PATH = PROCESSED_DATA_DIR / "sales_data_clean.csv"
CUSTOMERS_CLEAN_PATH = PROCESSED_DATA_DIR / "customers_clean.csv"


def print_section(title: str) -> None:
    """Print a visually distinct section header to keep console output readable."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
