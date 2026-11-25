"""Data loading, cleaning, validation, and grade processing module.

All operations work on copies of the input DataFrame so the original CSV
dataset is never modified during analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from data_generator import PASS_MARKS, calculate_grade, calculate_result

REQUIRED_COLUMNS: tuple[str, ...] = (
    "Student_ID",
    "Name",
    "Gender",
    "Age",
    "Attendance_Percentage",
    "Study_Hours",
    "Previous_Marks",
    "Assignment_Marks",
    "Final_Marks",
    "Grade",
    "Result",
)

NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "Age": (16.0, 35.0),
    "Attendance_Percentage": (0.0, 100.0),
    "Study_Hours": (0.0, 16.0),
    "Previous_Marks": (0.0, 100.0),
    "Assignment_Marks": (0.0, 100.0),
    "Final_Marks": (0.0, 100.0),
}

VALID_GENDERS: set[str] = {"Male", "Female", "Other"}


@dataclass
class CleaningReport:
    """Structured summary of all cleaning and validation actions performed."""

    initial_rows: int = 0
    final_rows: int = 0
    missing_before: dict[str, int] = field(default_factory=dict)
    missing_after: dict[str, int] = field(default_factory=dict)
    total_missing_before: int = 0
    total_missing_after: int = 0
    duplicate_rows_removed: int = 0
    duplicate_ids_removed: int = 0
    invalid_values_fixed: int = 0
    grades_recalculated: int = 0
    is_valid: bool = True
    validation_messages: list[str] = field(default_factory=list)


def load_raw_dataset(csv_path: Path | str) -> pd.DataFrame:
    """Load the raw CSV dataset safely without altering the source file.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If the CSV file is empty or missing required columns.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset file is empty: {path}") from exc

    if df.empty:
        raise ValueError("Loaded dataset contains 0 rows.")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_cols)}")

    return df


def detect_missing_values(df: pd.DataFrame) -> dict[str, int]:
    """Return a mapping of column name to missing value count."""
    return {col: int(df[col].isna().sum()) for col in df.columns}


def validate_dataset(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate cleaned dataset integrity, schema, ranges, and consistency."""
    messages: list[str] = []

    if df.empty:
        return False, ["Dataset is empty after cleaning."]

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        messages.append(f"Missing required columns: {', '.join(missing_cols)}")

    null_counts = df[list(REQUIRED_COLUMNS)].isna().sum()
    cols_with_nulls = {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}
    if cols_with_nulls:
        messages.append(f"Remaining missing values found: {cols_with_nulls}")

    if df["Student_ID"].duplicated().any():
        dup_count = int(df["Student_ID"].duplicated().sum())
        messages.append(f"Found {dup_count} duplicate Student_ID entries.")

    for col, (low, high) in NUMERIC_BOUNDS.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            out_of_bounds = int(((df[col] < low) | (df[col] > high)).sum())
            if out_of_bounds > 0:
                messages.append(
                    f"Column '{col}' has {out_of_bounds} values outside [{low}, {high}]."
                )

    expected_results = df["Final_Marks"].apply(calculate_result)
    result_mismatch = int((df["Result"] != expected_results).sum())
    if result_mismatch > 0:
        messages.append(f"Found {result_mismatch} inconsistent Result values.")

    if not messages:
        messages.append("All schema, range, uniqueness, and grading checks passed.")
        return True, messages

    return False, messages


def clean_student_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Clean, validate, and standardize a copy of the student dataset.

    The input DataFrame is never mutated in place.
    """
    if raw_df is None or raw_df.empty:
        report = CleaningReport(
            is_valid=False,
            validation_messages=["Input dataset is empty."],
        )
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS)), report

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in raw_df.columns]
    if missing_cols:
        report = CleaningReport(
            initial_rows=len(raw_df),
            is_valid=False,
            validation_messages=[f"Missing required columns: {', '.join(missing_cols)}"],
        )
        return raw_df.copy(), report

    df = raw_df.copy()
    report = CleaningReport(initial_rows=len(df))
    report.missing_before = detect_missing_values(df)
    report.total_missing_before = sum(report.missing_before.values())

    # 1. Remove exact duplicate rows
    before_exact = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    report.duplicate_rows_removed = before_exact - len(df)

    # 2. Clean and standardize string columns
    df["Student_ID"] = df["Student_ID"].astype("string").str.strip()
    df = df.dropna(subset=["Student_ID"]).reset_index(drop=True)
    df = df[df["Student_ID"] != ""].reset_index(drop=True)

    # Remove duplicate Student_IDs keeping the first occurrence
    before_id_dedup = len(df)
    df = df.drop_duplicates(subset=["Student_ID"], keep="first").reset_index(drop=True)
    report.duplicate_ids_removed = before_id_dedup - len(df)

    df["Name"] = df["Name"].astype("string").str.strip().fillna("Unknown Student")
    df.loc[df["Name"] == "", "Name"] = "Unknown Student"

    # Standardize Gender
    df["Gender"] = df["Gender"].astype("string").str.strip().str.title()
    invalid_gender_mask = df["Gender"].isna() | (~df["Gender"].isin(VALID_GENDERS))
    # Count only non-null invalid strings as invalid values fixed
    non_null_invalid_gender = int((df["Gender"].notna() & (~df["Gender"].isin(VALID_GENDERS))).sum())
    report.invalid_values_fixed += non_null_invalid_gender
    gender_mode_series = df.loc[~invalid_gender_mask, "Gender"].mode()
    fallback_gender = str(gender_mode_series.iloc[0]) if not gender_mode_series.empty else "Other"
    df.loc[invalid_gender_mask, "Gender"] = fallback_gender

    # 3. Data type correction & invalid value handling for numeric columns
    for col, (low, high) in NUMERIC_BOUNDS.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        invalid_mask = df[col].notna() & ((df[col] < low) | (df[col] > high))
        invalid_count = int(invalid_mask.sum())
        if invalid_count > 0:
            report.invalid_values_fixed += invalid_count
            df.loc[invalid_mask, col] = np.nan

        valid_series = df[col].dropna()
        fill_value = float(valid_series.median()) if not valid_series.empty else (low + high) / 2.0
        if col == "Age":
            fill_value = float(round(fill_value))
            df[col] = df[col].fillna(fill_value).round().astype(int)
        else:
            df[col] = df[col].fillna(round(fill_value, 1)).round(1)

    # 4. Consistent Grade & Pass/Fail calculation
    prev_grades = df["Grade"].astype("string").fillna("")
    prev_results = df["Result"].astype("string").fillna("")

    df["Grade"] = df["Final_Marks"].apply(calculate_grade)
    df["Result"] = df["Final_Marks"].apply(calculate_result)

    changed_grades = int(((prev_grades != df["Grade"]) | (prev_results != df["Result"])).sum())
    report.grades_recalculated = changed_grades

    # Convert string columns cleanly to standard Python string/object for broad compatibility
    for str_col in ("Student_ID", "Name", "Gender", "Grade", "Result"):
        df[str_col] = df[str_col].astype(str)

    report.final_rows = len(df)
    report.missing_after = detect_missing_values(df)
    report.total_missing_after = sum(report.missing_after.values())
    report.is_valid, report.validation_messages = validate_dataset(df)

    return df, report


def load_and_clean_dataset(csv_path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Convenience helper returning (raw_df, cleaned_df, cleaning_report)."""
    raw_df = load_raw_dataset(csv_path)
    cleaned_df, report = clean_student_data(raw_df)
    return raw_df, cleaned_df, report


def format_cleaning_summary(report: CleaningReport) -> dict[str, Any]:
    """Return a concise dictionary representation of the CleaningReport."""
    return {
        "Initial Rows": report.initial_rows,
        "Final Clean Rows": report.final_rows,
        "Total Missing Values Before": report.total_missing_before,
        "Total Missing Values After": report.total_missing_after,
        "Duplicate Rows Removed": report.duplicate_rows_removed,
        "Duplicate IDs Removed": report.duplicate_ids_removed,
        "Invalid Values Fixed": report.invalid_values_fixed,
        "Grades/Results Verified & Recalculated": report.grades_recalculated,
        "Validation Passed": report.is_valid,
    }
