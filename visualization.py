"""Data visualization module using Matplotlib and Seaborn.

Provides 8 consistent, publication-ready charts styled in a professional
blue-and-white theme for the Student Performance Analysis System.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Safe non-interactive backend for Streamlit & headless tests

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns

from analysis import GRADE_ORDER, get_correlation_matrix, get_grade_distribution, get_top_students
from data_generator import PASS_MARKS

# Theme Palette
PRIMARY_NAVY = "#1E3A8A"
PRIMARY_BLUE = "#2563EB"
SKY_BLUE = "#60A5FA"
LIGHT_BLUE = "#DBEAFE"
PASS_COLOR = "#2563EB"
FAIL_COLOR = "#EF4444"
TEXT_COLOR = "#0F172A"
GRID_COLOR = "#E2E8F0"

sns.set_theme(style="whitegrid", font_scale=1.0)


def _create_empty_figure(title: str, message: str = "No matching records available.") -> Figure:
    """Return a clean placeholder figure when filtered data is empty."""
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    ax.set_facecolor("white")
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="#64748B")
    ax.set_title(title, fontsize=13, fontweight="bold", color=PRIMARY_NAVY, pad=12)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _style_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent blue-and-white styling to Matplotlib axes."""
    ax.set_facecolor("white")
    ax.set_title(title, fontsize=13, fontweight="bold", color=PRIMARY_NAVY, pad=12)
    ax.set_xlabel(xlabel, fontsize=10.5, fontweight="bold", color=TEXT_COLOR, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold", color=TEXT_COLOR, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.6, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")


def plot_marks_distribution(df: pd.DataFrame) -> Figure:
    """1. Marks distribution histogram with KDE, pass line, and mean line."""
    title = "Distribution of Final Marks"
    if df is None or df.empty:
        return _create_empty_figure(title)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor="white")
    sns.histplot(
        data=df,
        x="Final_Marks",
        bins=20,
        kde=True,
        color=PRIMARY_BLUE,
        edgecolor="white",
        alpha=0.82,
        ax=ax,
    )
    mean_val = float(df["Final_Marks"].mean())
    ax.axvline(PASS_MARKS, color=FAIL_COLOR, linestyle="--", linewidth=2, label=f"Pass Threshold ({int(PASS_MARKS)})")
    ax.axvline(mean_val, color=PRIMARY_NAVY, linestyle="-.", linewidth=2, label=f"Mean Marks ({mean_val:.1f})")

    _style_axis(ax, title, "Final Marks (0 - 100)", "Number of Students")
    ax.legend(frameon=True, facecolor="white", edgecolor=GRID_COLOR)
    fig.tight_layout()
    return fig


def plot_grade_distribution(df: pd.DataFrame) -> Figure:
    """2. Grade distribution bar chart across A+ to F."""
    title = "Grade Distribution"
    if df is None or df.empty:
        return _create_empty_figure(title)

    grade_df = get_grade_distribution(df)
    palette = ["#1E3A8A", "#1D4ED8", "#2563EB", "#3B82F6", "#60A5FA", "#EF4444"]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor="white")
    bars = ax.bar(grade_df["Grade"], grade_df["Count"], color=palette, edgecolor="white", width=0.6)

    for bar, (_, row) in zip(bars, grade_df.iterrows()):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(grade_df["Count"].max() * 0.02, 0.3),
            f"{int(row['Count'])}\n({row['Percentage']:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT_COLOR,
            fontweight="bold",
        )

    max_count = int(grade_df["Count"].max()) if not grade_df.empty else 10
    ax.set_ylim(0, max(max_count * 1.2, 5))
    _style_axis(ax, title, "Grade", "Number of Students")
    fig.tight_layout()
    return fig


def plot_gender_average_marks(df: pd.DataFrame) -> Figure:
    """3. Gender-wise average final marks bar chart."""
    title = "Gender-Wise Average Final Marks"
    if df is None or df.empty:
        return _create_empty_figure(title)

    summary = (
        df.groupby("Gender", as_index=False)
        .agg(Avg_Marks=("Final_Marks", "mean"), Count=("Student_ID", "count"))
        .sort_values("Avg_Marks", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.6), facecolor="white")
    colors = [PRIMARY_NAVY, PRIMARY_BLUE, SKY_BLUE][: len(summary)]
    bars = ax.bar(summary["Gender"], summary["Avg_Marks"], color=colors, width=0.5, edgecolor="white")

    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.5,
            f"{row['Avg_Marks']:.1f}\n(n={int(row['Count'])})",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=TEXT_COLOR,
        )

    ax.axhline(PASS_MARKS, color=FAIL_COLOR, linestyle="--", linewidth=1.5, label=f"Pass Marks ({int(PASS_MARKS)})")
    ax.set_ylim(0, 108)
    _style_axis(ax, title, "Gender", "Average Final Marks")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=GRID_COLOR)
    fig.tight_layout()
    return fig


def plot_attendance_vs_marks(df: pd.DataFrame) -> Figure:
    """4. Attendance percentage vs final marks scatter plot with trendline."""
    title = "Attendance Percentage vs Final Marks"
    if df is None or df.empty:
        return _create_empty_figure(title)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor="white")
    palette = {"Pass": PASS_COLOR, "Fail": FAIL_COLOR}
    sns.scatterplot(
        data=df,
        x="Attendance_Percentage",
        y="Final_Marks",
        hue="Result",
        palette=palette,
        alpha=0.78,
        s=55,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    if len(df) >= 2 and df["Attendance_Percentage"].std() > 0:
        sns.regplot(
            data=df,
            x="Attendance_Percentage",
            y="Final_Marks",
            scatter=False,
            color=PRIMARY_NAVY,
            line_kws={"linewidth": 2, "linestyle": "--", "label": "Trendline"},
            ax=ax,
        )

    ax.axhline(PASS_MARKS, color="#64748B", linestyle=":", linewidth=1.4, label=f"Pass Threshold ({int(PASS_MARKS)})")
    _style_axis(ax, title, "Attendance Percentage (%)", "Final Marks")
    ax.legend(title="Outcome", frameon=True, facecolor="white", edgecolor=GRID_COLOR)
    fig.tight_layout()
    return fig


def plot_study_hours_vs_marks(df: pd.DataFrame) -> Figure:
    """5. Study hours vs final marks scatter plot with trendline."""
    title = "Daily Study Hours vs Final Marks"
    if df is None or df.empty:
        return _create_empty_figure(title)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor="white")
    palette = {"Pass": PASS_COLOR, "Fail": FAIL_COLOR}
    sns.scatterplot(
        data=df,
        x="Study_Hours",
        y="Final_Marks",
        hue="Result",
        palette=palette,
        alpha=0.78,
        s=55,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    if len(df) >= 2 and df["Study_Hours"].std() > 0:
        sns.regplot(
            data=df,
            x="Study_Hours",
            y="Final_Marks",
            scatter=False,
            color=PRIMARY_NAVY,
            line_kws={"linewidth": 2, "linestyle": "--", "label": "Trendline"},
            ax=ax,
        )

    ax.axhline(PASS_MARKS, color="#64748B", linestyle=":", linewidth=1.4, label=f"Pass Threshold ({int(PASS_MARKS)})")
    _style_axis(ax, title, "Daily Study Hours", "Final Marks")
    ax.legend(title="Outcome", frameon=True, facecolor="white", edgecolor=GRID_COLOR)
    fig.tight_layout()
    return fig


def plot_top_10_students(df: pd.DataFrame) -> Figure:
    """6. Top 10 students horizontal bar chart."""
    title = "Top 10 Performing Students"
    top_df = get_top_students(df, n=10)
    if top_df.empty:
        return _create_empty_figure(title)

    plot_df = top_df.iloc[::-1].copy()
    plot_df["Label"] = plot_df["Name"] + " (" + plot_df["Student_ID"] + ")"

    fig, ax = plt.subplots(figsize=(9.0, 5.2), facecolor="white")
    colors = sns.color_palette("Blues", n_colors=len(plot_df))
    bars = ax.barh(plot_df["Label"], plot_df["Final_Marks"], color=colors, edgecolor="white", height=0.65)

    for bar, val in zip(bars, plot_df["Final_Marks"]):
        ax.text(
            val + 0.8,
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.1f}",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color=PRIMARY_NAVY,
        )

    ax.set_xlim(0, 108)
    _style_axis(ax, title, "Final Marks", "Student")
    fig.tight_layout()
    return fig


def plot_pass_vs_fail(df: pd.DataFrame) -> Figure:
    """7. Pass vs fail pie chart."""
    title = "Pass vs Fail Ratio"
    if df is None or df.empty:
        return _create_empty_figure(title)

    pass_count = int((df["Result"] == "Pass").sum())
    fail_count = int((df["Result"] == "Fail").sum())
    if pass_count + fail_count == 0:
        return _create_empty_figure(title)

    labels = ["Pass", "Fail"]
    sizes = [pass_count, fail_count]
    colors = [PRIMARY_BLUE, FAIL_COLOR]

    fig, ax = plt.subplots(figsize=(6.5, 4.8), facecolor="white")
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=[f"Pass ({pass_count})", f"Fail ({fail_count})"],
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        explode=(0.03, 0.03),
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 10.5, "color": TEXT_COLOR, "fontweight": "bold"},
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    ax.set_title(title, fontsize=13, fontweight="bold", color=PRIMARY_NAVY, pad=12)
    ax.legend(wedges, labels, title="Result", loc="lower right", frameon=True, facecolor="white")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> Figure:
    """8. Correlation heatmap of numeric student performance variables."""
    title = "Academic Metrics Correlation Heatmap"
    corr_matrix = get_correlation_matrix(df)
    if corr_matrix.empty:
        return _create_empty_figure(title, "Insufficient numeric data to compute correlations.")

    readable_labels = {
        "Age": "Age",
        "Attendance_Percentage": "Attendance %",
        "Study_Hours": "Study Hours",
        "Previous_Marks": "Previous Marks",
        "Assignment_Marks": "Assignment Marks",
        "Final_Marks": "Final Marks",
    }
    renamed = corr_matrix.rename(index=readable_labels, columns=readable_labels)

    fig, ax = plt.subplots(figsize=(8.2, 6.0), facecolor="white")
    sns.heatmap(
        renamed,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=-0.2,
        vmax=1.0,
        linewidths=0.8,
        linecolor="white",
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", color=PRIMARY_NAVY, pad=14)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9.5)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9.5)
    fig.tight_layout()
    return fig
