"""Exploratory Data Analysis (EDA) and automated reporting module.

Provides safe, division-by-zero-free analytics functions over cleaned student
DataFrames without modifying the underlying dataset.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from data_cleaning import CleaningReport
from data_generator import PASS_MARKS

GRADE_ORDER: list[str] = ["A+", "A", "B", "C", "D", "F"]
NUMERIC_ANALYSIS_COLS: list[str] = [
    "Age",
    "Attendance_Percentage",
    "Study_Hours",
    "Previous_Marks",
    "Assignment_Marks",
    "Final_Marks",
]


def _safe_corr(series_x: pd.Series, series_y: pd.Series) -> float:
    """Compute Pearson correlation safely without NaN/zero-variance errors."""
    valid = pd.concat([series_x, series_y], axis=1).dropna()
    if len(valid) < 2:
        return 0.0
    if valid.iloc[:, 0].std() == 0 or valid.iloc[:, 1].std() == 0:
        return 0.0
    corr = float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))
    return 0.0 if np.isnan(corr) else round(corr, 4)


def get_overview_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate core KPI summary metrics safely for empty or populated data."""
    if df is None or df.empty:
        return {
            "total_students": 0,
            "avg_marks": 0.0,
            "median_marks": 0.0,
            "std_marks": 0.0,
            "max_marks": 0.0,
            "min_marks": 0.0,
            "pass_count": 0,
            "fail_count": 0,
            "pass_percentage": 0.0,
            "fail_percentage": 0.0,
            "avg_attendance": 0.0,
            "avg_study_hours": 0.0,
            "avg_previous_marks": 0.0,
            "avg_assignment_marks": 0.0,
        }

    total = len(df)
    pass_count = int((df["Result"] == "Pass").sum())
    fail_count = total - pass_count
    pass_pct = round((pass_count / total) * 100.0, 2) if total > 0 else 0.0
    fail_pct = round((fail_count / total) * 100.0, 2) if total > 0 else 0.0

    return {
        "total_students": total,
        "avg_marks": round(float(df["Final_Marks"].mean()), 2),
        "median_marks": round(float(df["Final_Marks"].median()), 2),
        "std_marks": round(float(df["Final_Marks"].std(ddof=1)), 2) if total > 1 else 0.0,
        "max_marks": round(float(df["Final_Marks"].max()), 2),
        "min_marks": round(float(df["Final_Marks"].min()), 2),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_percentage": pass_pct,
        "fail_percentage": fail_pct,
        "avg_attendance": round(float(df["Attendance_Percentage"].mean()), 2),
        "avg_study_hours": round(float(df["Study_Hours"].mean()), 2),
        "avg_previous_marks": round(float(df["Previous_Marks"].mean()), 2),
        "avg_assignment_marks": round(float(df["Assignment_Marks"].mean()), 2),
    }


def get_gender_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Return gender-wise performance breakdown."""
    columns = [
        "Gender",
        "Student_Count",
        "Average_Marks",
        "Median_Marks",
        "Max_Marks",
        "Min_Marks",
        "Average_Attendance",
        "Average_Study_Hours",
        "Pass_Rate_Pct",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    work = df.copy()
    work["Passed"] = (work["Result"] == "Pass").astype(float) * 100.0

    grouped = (
        work.groupby("Gender", as_index=False)
        .agg(
            Student_Count=("Student_ID", "count"),
            Average_Marks=("Final_Marks", "mean"),
            Median_Marks=("Final_Marks", "median"),
            Max_Marks=("Final_Marks", "max"),
            Min_Marks=("Final_Marks", "min"),
            Average_Attendance=("Attendance_Percentage", "mean"),
            Average_Study_Hours=("Study_Hours", "mean"),
            Pass_Rate_Pct=("Passed", "mean"),
        )
        .sort_values("Average_Marks", ascending=False)
        .reset_index(drop=True)
    )

    numeric_cols = [col for col in grouped.columns if col not in ("Gender", "Student_Count")]
    for col in numeric_cols:
        grouped[col] = grouped[col].round(2)

    return grouped


def get_grade_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return grade counts and percentages ordered from A+ to F."""
    if df is None or df.empty:
        return pd.DataFrame({"Grade": GRADE_ORDER, "Count": [0] * len(GRADE_ORDER), "Percentage": [0.0] * len(GRADE_ORDER)})

    total = len(df)
    counts = df["Grade"].value_counts().reindex(GRADE_ORDER, fill_value=0)
    percentages = (counts / total * 100.0).round(2) if total > 0 else counts.astype(float)

    return pd.DataFrame(
        {
            "Grade": GRADE_ORDER,
            "Count": counts.astype(int).values,
            "Percentage": percentages.values,
        }
    )


def get_top_students(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top N students ranked by Final_Marks, Assignment_Marks, and Attendance."""
    if df is None or df.empty:
        return pd.DataFrame()
    return (
        df.sort_values(
            by=["Final_Marks", "Assignment_Marks", "Attendance_Percentage"],
            ascending=[False, False, False],
        )
        .head(n)
        .reset_index(drop=True)
    )


def get_lowest_students(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return bottom N lowest-performing students ranked by Final_Marks ascending."""
    if df is None or df.empty:
        return pd.DataFrame()
    return (
        df.sort_values(
            by=["Final_Marks", "Attendance_Percentage", "Study_Hours"],
            ascending=[True, True, True],
        )
        .head(n)
        .reset_index(drop=True)
    )


def get_attendance_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze attendance distribution, brackets, and correlation with marks."""
    bracket_cols = ["Attendance_Bracket", "Student_Count", "Average_Marks", "Pass_Rate_Pct"]
    if df is None or df.empty:
        return {
            "avg_attendance": 0.0,
            "min_attendance": 0.0,
            "max_attendance": 0.0,
            "median_attendance": 0.0,
            "correlation_with_marks": 0.0,
            "below_75_count": 0,
            "below_75_pass_rate": 0.0,
            "above_75_pass_rate": 0.0,
            "bracket_summary": pd.DataFrame(columns=bracket_cols),
        }

    work = df.copy()
    work["Passed"] = (work["Result"] == "Pass").astype(float) * 100.0
    bins = [0, 60, 75, 90, 100.01]
    labels = ["Below 60% (Low)", "60% - 74.9% (Moderate)", "75% - 89.9% (Good)", "90% - 100% (Excellent)"]
    work["Attendance_Bracket"] = pd.cut(
        work["Attendance_Percentage"], bins=bins, labels=labels, right=False
    )

    bracket_summary = (
        work.groupby("Attendance_Bracket", observed=False, as_index=False)
        .agg(
            Student_Count=("Student_ID", "count"),
            Average_Marks=("Final_Marks", "mean"),
            Pass_Rate_Pct=("Passed", "mean"),
        )
    )
    bracket_summary["Average_Marks"] = bracket_summary["Average_Marks"].fillna(0.0).round(2)
    bracket_summary["Pass_Rate_Pct"] = bracket_summary["Pass_Rate_Pct"].fillna(0.0).round(2)

    below_75 = work[work["Attendance_Percentage"] < 75.0]
    above_75 = work[work["Attendance_Percentage"] >= 75.0]

    return {
        "avg_attendance": round(float(work["Attendance_Percentage"].mean()), 2),
        "min_attendance": round(float(work["Attendance_Percentage"].min()), 2),
        "max_attendance": round(float(work["Attendance_Percentage"].max()), 2),
        "median_attendance": round(float(work["Attendance_Percentage"].median()), 2),
        "correlation_with_marks": _safe_corr(work["Attendance_Percentage"], work["Final_Marks"]),
        "below_75_count": len(below_75),
        "below_75_pass_rate": round(float(below_75["Passed"].mean()), 2) if not below_75.empty else 0.0,
        "above_75_pass_rate": round(float(above_75["Passed"].mean()), 2) if not above_75.empty else 0.0,
        "bracket_summary": bracket_summary,
    }


def get_study_hours_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze daily study-hours distribution, brackets, and correlation with marks."""
    bracket_cols = ["Study_Hours_Bracket", "Student_Count", "Average_Marks", "Pass_Rate_Pct"]
    if df is None or df.empty:
        return {
            "avg_study_hours": 0.0,
            "min_study_hours": 0.0,
            "max_study_hours": 0.0,
            "median_study_hours": 0.0,
            "correlation_with_marks": 0.0,
            "bracket_summary": pd.DataFrame(columns=bracket_cols),
        }

    work = df.copy()
    work["Passed"] = (work["Result"] == "Pass").astype(float) * 100.0
    bins = [0, 2.0, 4.0, 6.0, 24.0]
    labels = ["< 2 hrs/day", "2 - 3.9 hrs/day", "4 - 5.9 hrs/day", "6+ hrs/day"]
    work["Study_Hours_Bracket"] = pd.cut(
        work["Study_Hours"], bins=bins, labels=labels, right=False
    )

    bracket_summary = (
        work.groupby("Study_Hours_Bracket", observed=False, as_index=False)
        .agg(
            Student_Count=("Student_ID", "count"),
            Average_Marks=("Final_Marks", "mean"),
            Pass_Rate_Pct=("Passed", "mean"),
        )
    )
    bracket_summary["Average_Marks"] = bracket_summary["Average_Marks"].fillna(0.0).round(2)
    bracket_summary["Pass_Rate_Pct"] = bracket_summary["Pass_Rate_Pct"].fillna(0.0).round(2)

    return {
        "avg_study_hours": round(float(work["Study_Hours"].mean()), 2),
        "min_study_hours": round(float(work["Study_Hours"].min()), 2),
        "max_study_hours": round(float(work["Study_Hours"].max()), 2),
        "median_study_hours": round(float(work["Study_Hours"].median()), 2),
        "correlation_with_marks": _safe_corr(work["Study_Hours"], work["Final_Marks"]),
        "bracket_summary": bracket_summary,
    }


def get_performance_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Group students into overall academic performance tiers."""
    labels = [
        "Fail (< 40)",
        "Average (40 - 59.9)",
        "Good (60 - 74.9)",
        "Very Good (75 - 89.9)",
        "Outstanding (90 - 100)",
    ]
    if df is None or df.empty:
        return pd.DataFrame({"Performance_Tier": labels, "Count": [0] * len(labels), "Percentage": [0.0] * len(labels)})

    bins = [0, 40, 60, 75, 90, 100.01]
    tiers = pd.cut(df["Final_Marks"], bins=bins, labels=labels, right=False)
    counts = tiers.value_counts().reindex(labels, fill_value=0)
    total = len(df)
    percentages = (counts / total * 100.0).round(2) if total > 0 else counts.astype(float)

    return pd.DataFrame(
        {
            "Performance_Tier": labels,
            "Count": counts.astype(int).values,
            "Percentage": percentages.values,
        }
    )


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for numeric student attributes."""
    if df is None or df.empty:
        return pd.DataFrame()
    available = [col for col in NUMERIC_ANALYSIS_COLS if col in df.columns]
    if len(available) < 2:
        return pd.DataFrame()
    return df[available].corr().round(3)


def generate_insights(df: pd.DataFrame) -> list[str]:
    """Produce data-driven academic conclusions from the EDA results."""
    if df is None or df.empty:
        return ["No student records are available to generate insights."]

    overview = get_overview_metrics(df)
    gender_df = get_gender_performance(df)
    grade_df = get_grade_distribution(df)
    att = get_attendance_analysis(df)
    study = get_study_hours_analysis(df)

    insights: list[str] = []

    # 1. Overall pass & marks insight
    insights.append(
        f"Out of {overview['total_students']} students, {overview['pass_count']} passed "
        f"({overview['pass_percentage']:.1f}%) and {overview['fail_count']} failed "
        f"({overview['fail_percentage']:.1f}%) based on the {int(PASS_MARKS)}-mark passing threshold."
    )
    insights.append(
        f"Final marks average {overview['avg_marks']:.2f} (median {overview['median_marks']:.2f}), "
        f"ranging from a minimum of {overview['min_marks']:.1f} to a maximum of {overview['max_marks']:.1f}."
    )

    # 2. Dominant grade insight
    if not grade_df.empty and grade_df["Count"].sum() > 0:
        top_grade_row = grade_df.loc[grade_df["Count"].idxmax()]
        insights.append(
            f"Grade '{top_grade_row['Grade']}' is the largest cohort with {int(top_grade_row['Count'])} students "
            f"({top_grade_row['Percentage']:.1f}% of the batch)."
        )

    # 3. Attendance impact insight
    att_corr = att["correlation_with_marks"]
    att_strength = "strong" if abs(att_corr) >= 0.6 else ("moderate" if abs(att_corr) >= 0.35 else "weak")
    insights.append(
        f"Attendance shows a {att_strength} positive correlation (r = {att_corr:.3f}) with Final Marks. "
        f"Students with >=75% attendance achieved a {att['above_75_pass_rate']:.1f}% pass rate, "
        f"compared to {att['below_75_pass_rate']:.1f}% for students below 75%."
    )

    # 4. Study hours impact insight
    study_corr = study["correlation_with_marks"]
    study_strength = "strong" if abs(study_corr) >= 0.6 else ("moderate" if abs(study_corr) >= 0.35 else "weak")
    insights.append(
        f"Daily study hours exhibit a {study_strength} positive correlation (r = {study_corr:.3f}) with Final Marks, "
        f"averaging {study['avg_study_hours']:.2f} hours/day across all students."
    )

    # 5. Gender comparison insight
    if len(gender_df) >= 2:
        highest_g = gender_df.iloc[0]
        lowest_g = gender_df.iloc[-1]
        gap = round(float(highest_g["Average_Marks"] - lowest_g["Average_Marks"]), 2)
        insights.append(
            f"Gender-wise average marks are led by {highest_g['Gender']} ({highest_g['Average_Marks']:.2f}) "
            f"and followed by {lowest_g['Gender']} ({lowest_g['Average_Marks']:.2f}), a spread of {gap:.2f} marks."
        )

    # 6. Correlation with prior/assignment marks
    prev_corr = _safe_corr(df["Previous_Marks"], df["Final_Marks"])
    assign_corr = _safe_corr(df["Assignment_Marks"], df["Final_Marks"])
    insights.append(
        f"Previous semester marks (r = {prev_corr:.3f}) and assignment performance (r = {assign_corr:.3f}) "
        f"are reliable predictors of final examination outcomes."
    )

    return insights


def run_full_eda(df: pd.DataFrame) -> dict[str, Any]:
    """Execute all EDA computations and return a consolidated results dictionary."""
    return {
        "overview": get_overview_metrics(df),
        "gender_performance": get_gender_performance(df),
        "grade_distribution": get_grade_distribution(df),
        "performance_distribution": get_performance_distribution(df),
        "attendance_analysis": get_attendance_analysis(df),
        "study_analysis": get_study_hours_analysis(df),
        "top_10_students": get_top_students(df, n=10),
        "lowest_students": get_lowest_students(df, n=10),
        "correlation_matrix": get_correlation_matrix(df),
        "insights": generate_insights(df),
    }


def generate_text_report(
    df: pd.DataFrame,
    cleaning_report: CleaningReport | None = None,
    output_path: Path | str | None = None,
) -> str:
    """Create a formatted academic analysis report string and optionally save to disk."""
    eda = run_full_eda(df)
    overview = eda["overview"]
    gender_df = eda["gender_performance"]
    grade_df = eda["grade_distribution"]
    perf_df = eda["performance_distribution"]
    att = eda["attendance_analysis"]
    study = eda["study_analysis"]
    top_10 = eda["top_10_students"]
    lowest = eda["lowest_students"]

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "=" * 78,
        "STUDENT PERFORMANCE ANALYSIS SYSTEM - ACADEMIC EDA REPORT",
        "=" * 78,
        f"Generated On : {timestamp}",
        f"Pass Marks   : {int(PASS_MARKS)} / 100",
        "",
    ]

    if cleaning_report is not None:
        lines.extend(
            [
                "1. DATA CLEANING & VALIDATION SUMMARY",
                "-" * 78,
                f"Initial Rows                  : {cleaning_report.initial_rows}",
                f"Final Clean Rows              : {cleaning_report.final_rows}",
                f"Missing Values Detected       : {cleaning_report.total_missing_before}",
                f"Missing Values After Cleaning : {cleaning_report.total_missing_after}",
                f"Duplicate Rows Removed        : {cleaning_report.duplicate_rows_removed}",
                f"Duplicate IDs Removed         : {cleaning_report.duplicate_ids_removed}",
                f"Invalid Values Fixed          : {cleaning_report.invalid_values_fixed}",
                f"Grades/Results Verified       : {cleaning_report.grades_recalculated}",
                f"Validation Status             : {'PASSED' if cleaning_report.is_valid else 'FAILED'}",
                "",
            ]
        )

    lines.extend(
        [
            "2. OVERALL PERFORMANCE OVERVIEW",
            "-" * 78,
            f"Total Students         : {overview['total_students']}",
            f"Average Final Marks    : {overview['avg_marks']:.2f}",
            f"Median Final Marks     : {overview['median_marks']:.2f}",
            f"Maximum Final Marks    : {overview['max_marks']:.2f}",
            f"Minimum Final Marks    : {overview['min_marks']:.2f}",
            f"Pass Count / Rate      : {overview['pass_count']} ({overview['pass_percentage']:.2f}%)",
            f"Fail Count / Rate      : {overview['fail_count']} ({overview['fail_percentage']:.2f}%)",
            f"Average Attendance     : {overview['avg_attendance']:.2f}%",
            f"Average Study Hours    : {overview['avg_study_hours']:.2f} hrs/day",
            "",
            "3. GRADE DISTRIBUTION",
            "-" * 78,
        ]
    )
    for _, row in grade_df.iterrows():
        lines.append(f"Grade {row['Grade']:<3} : {int(row['Count']):>4} students ({row['Percentage']:>6.2f}%)")

    lines.extend(["", "4. PERFORMANCE TIER DISTRIBUTION", "-" * 78])
    for _, row in perf_df.iterrows():
        lines.append(
            f"{row['Performance_Tier']:<25} : {int(row['Count']):>4} students ({row['Percentage']:>6.2f}%)"
        )

    lines.extend(["", "5. GENDER-WISE PERFORMANCE", "-" * 78])
    if gender_df.empty:
        lines.append("No gender breakdown available.")
    else:
        for _, row in gender_df.iterrows():
            lines.append(
                f"{row['Gender']:<8} | Count: {int(row['Student_Count']):>3} | "
                f"Avg Marks: {row['Average_Marks']:>6.2f} | "
                f"Pass Rate: {row['Pass_Rate_Pct']:>6.2f}% | "
                f"Avg Attendance: {row['Average_Attendance']:>6.2f}%"
            )

    lines.extend(
        [
            "",
            "6. ATTENDANCE & STUDY HOURS CORRELATION ANALYSIS",
            "-" * 78,
            f"Attendance vs Final Marks Correlation  : r = {att['correlation_with_marks']:.4f}",
            f"Pass Rate (Attendance >= 75%)          : {att['above_75_pass_rate']:.2f}%",
            f"Pass Rate (Attendance <  75%)          : {att['below_75_pass_rate']:.2f}%",
            f"Study Hours vs Final Marks Correlation : r = {study['correlation_with_marks']:.4f}",
            "",
            "7. TOP 10 PERFORMING STUDENTS",
            "-" * 78,
        ]
    )
    if top_10.empty:
        lines.append("No student records available.")
    else:
        for idx, (_, row) in enumerate(top_10.iterrows(), start=1):
            lines.append(
                f"{idx:>2}. {row['Student_ID']} | {row['Name']:<20} | "
                f"Marks: {row['Final_Marks']:>5.1f} | Grade: {row['Grade']:<2} | "
                f"Attendance: {row['Attendance_Percentage']:>5.1f}%"
            )

    lines.extend(["", "8. LOWEST-PERFORMING STUDENTS (AT-RISK COHORT)", "-" * 78])
    if lowest.empty:
        lines.append("No student records available.")
    else:
        for idx, (_, row) in enumerate(lowest.iterrows(), start=1):
            lines.append(
                f"{idx:>2}. {row['Student_ID']} | {row['Name']:<20} | "
                f"Marks: {row['Final_Marks']:>5.1f} | Grade: {row['Grade']:<2} | "
                f"Attendance: {row['Attendance_Percentage']:>5.1f}% | Result: {row['Result']}"
            )

    lines.extend(["", "9. KEY CONCLUSIONS & RECOMMENDATIONS", "-" * 78])
    for idx, insight in enumerate(eda["insights"], start=1):
        lines.append(f"{idx}. {insight}")

    lines.extend(["", "=" * 78, "END OF REPORT", "=" * 78])
    report_text = "\n".join(lines)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report_text, encoding="utf-8")

    return report_text
