"""
data_quality.py

Runs a concise data-quality check on the raw dataset and writes a plain-text
report to outputs/metrics/data_quality_report.txt. This documents missing
values and other quality issues before any cleaning or modeling happens,
as required by the Week 3 spec.
"""

from pathlib import Path

import pandas as pd

from src.data_loader import TARGET_COLUMN, load_raw_data

REPORT_PATH = Path("outputs/metrics/data_quality_report.txt")


def build_quality_report(df: pd.DataFrame) -> str:
    """Build a plain-English data-quality report as a single string."""
    lines = []
    lines.append("DATA QUALITY REPORT")
    lines.append("=" * 40)
    lines.append(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    lines.append("")

    lines.append("Column dtypes:")
    for col, dtype in df.dtypes.items():
        lines.append(f"  - {col}: {dtype}")
    lines.append("")

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    lines.append("Missing values (columns with at least one missing value):")
    if missing.empty:
        lines.append("  - None found.")
    else:
        for col, count in missing.items():
            pct = 100 * count / len(df)
            lines.append(f"  - {col}: {count} missing ({pct:.1f}%)")
    lines.append("")

    duplicate_count = df.duplicated().sum()
    lines.append(f"Duplicate rows: {duplicate_count}")
    lines.append("")

    lines.append(f"Target column: '{TARGET_COLUMN}'")
    class_counts = df[TARGET_COLUMN].value_counts()
    lines.append("Target class balance:")
    for value, count in class_counts.items():
        pct = 100 * count / len(df)
        lines.append(f"  - {value}: {count} ({pct:.1f}%)")

    return "\n".join(lines)


def run_data_quality_check() -> None:
    """Load the raw data, build the report, print it, and save it to disk."""
    df = load_raw_data()
    report = build_quality_report(df)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    run_data_quality_check()
