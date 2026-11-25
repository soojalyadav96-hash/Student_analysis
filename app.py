"""Streamlit Dashboard for the Student Performance Analysis System.

Provides interactive exploration of student academic performance, attendance,
study habits, data cleaning diagnostics, and downloadable analytical reports.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from analysis import (
    GRADE_ORDER,
    generate_insights,
    generate_text_report,
    get_attendance_analysis,
    get_correlation_matrix,
    get_gender_performance,
    get_grade_distribution,
    get_lowest_students,
    get_overview_metrics,
    get_performance_distribution,
    get_study_hours_analysis,
    get_top_students,
)
from data_cleaning import (
    CleaningReport,
    format_cleaning_summary,
    load_and_clean_dataset,
)
from data_generator import PASS_MARKS, save_dataset
from visualization import (
    PRIMARY_BLUE,
    PRIMARY_NAVY,
    plot_attendance_vs_marks,
    plot_correlation_heatmap,
    plot_gender_average_marks,
    plot_grade_distribution,
    plot_marks_distribution,
    plot_pass_vs_fail,
    plot_study_hours_vs_marks,
    plot_top_10_students,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "students.csv"
REPORT_PATH = BASE_DIR / "reports" / "analysis_report.txt"

st.set_page_config(
    page_title="Student Performance Analysis System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_custom_theme() -> None:
    """Inject professional blue-and-white CSS styling."""
    st.markdown(
        """
        <style>
        /* Main app background */
        .stApp {
            background-color: #F8FAFC;
            color: #0F172A;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F295C 0%, #1E3A8A 100%);
            color: #FFFFFF;
        }
        [data-testid="stSidebar"] * {
            color: #FFFFFF !important;
        }
        [data-testid="stSidebar"] .stRadio > div {
            gap: 0.35rem;
        }
        [data-testid="stSidebar"] [data-baseweb="radio"] {
            background-color: rgba(255, 255, 255, 0.08);
            border-radius: 8px;
            padding: 0.45rem 0.65rem;
        }
        [data-testid="stSidebar"] [data-baseweb="radio"]:hover {
            background-color: rgba(255, 255, 255, 0.16);
        }

        /* Header banner */
        .hero-banner {
            background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
            color: #FFFFFF;
            padding: 1.4rem 1.8rem;
            border-radius: 12px;
            margin-bottom: 1.25rem;
            box-shadow: 0 4px 14px rgba(30, 58, 138, 0.18);
        }
        .hero-banner h1 {
            margin: 0;
            font-size: 1.85rem;
            font-weight: 700;
            color: #FFFFFF;
        }
        .hero-banner p {
            margin: 0.35rem 0 0 0;
            font-size: 0.98rem;
            color: #DBEAFE;
        }

        /* KPI cards */
        .kpi-card {
            background-color: #FFFFFF;
            border: 1px solid #DBEAFE;
            border-top: 4px solid #2563EB;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
            margin-bottom: 0.75rem;
        }
        .kpi-label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }
        .kpi-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-top: 0.25rem;
        }
        .kpi-sub {
            font-size: 0.82rem;
            color: #64748B;
            margin-top: 0.2rem;
        }

        /* Section cards */
        .info-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 4px solid #2563EB;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
        }

        /* Headings */
        h2, h3 {
            color: #1E3A8A;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, subtitle: str = "") -> None:
    """Render a styled blue-and-white KPI card."""
    sub_html = f'<div class="kpi-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_figure(fig: plt.Figure) -> None:
    """Display a Matplotlib figure cleanly and release figure memory."""
    st.pyplot(fig, width="stretch")
    plt.close(fig)


@st.cache_data(show_spinner=False)
def get_cleaned_data(csv_path_str: str) -> tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Load and clean the CSV dataset without mutating the source file."""
    return load_and_clean_dataset(csv_path_str)


def render_overview_section(df: pd.DataFrame) -> None:
    """Section A: High-level KPIs, distribution charts, and summary."""
    st.subheader("A. Overview & Overall Performance Summary")
    st.write(
        "Key academic indicators computed from the cleaned student dataset "
        f"(pass threshold: {int(PASS_MARKS)} marks)."
    )

    metrics = get_overview_metrics(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi_card(
            "Total Students",
            f"{metrics['total_students']}",
            f"Pass: {metrics['pass_count']} | Fail: {metrics['fail_count']}",
        )
    with col2:
        render_kpi_card(
            "Average Marks",
            f"{metrics['avg_marks']:.2f}",
            f"Range: {metrics['min_marks']:.1f} - {metrics['max_marks']:.1f}",
        )
    with col3:
        render_kpi_card(
            "Pass Percentage",
            f"{metrics['pass_percentage']:.1f}%",
            f"Fail Rate: {metrics['fail_percentage']:.1f}%",
        )
    with col4:
        render_kpi_card(
            "Average Attendance",
            f"{metrics['avg_attendance']:.1f}%",
            f"Avg Study: {metrics['avg_study_hours']:.2f} hrs/day",
        )

    st.markdown("---")
    chart_col1, chart_col2 = st.columns([1.35, 1.0])
    with chart_col1:
        render_figure(plot_marks_distribution(df))
    with chart_col2:
        render_figure(plot_pass_vs_fail(df))

    st.markdown("### Performance Tier Breakdown & Key Takeaways")
    tier_col, insight_col = st.columns([1.0, 1.2])
    with tier_col:
        perf_df = get_performance_distribution(df)
        st.dataframe(perf_df, hide_index=True, width="stretch")
    with insight_col:
        insights = generate_insights(df)
        bullet_html = "".join(f"<li>{item}</li>" for item in insights[:4])
        st.markdown(
            f"""
            <div class="info-card">
                <strong>Executive Performance Highlights</strong>
                <ul style="margin-top:0.5rem; margin-bottom:0;">
                    {bullet_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_student_analysis_section(df: pd.DataFrame) -> None:
    """Section B: Gender analysis, grade distribution, top 10, and student lookup."""
    st.subheader("B. Student Performance Analysis")
    st.write("Explore demographic performance, grade cohorts, top achievers, and individual profiles.")

    col1, col2 = st.columns(2)
    with col1:
        render_figure(plot_gender_average_marks(df))
    with col2:
        render_figure(plot_grade_distribution(df))

    st.markdown("### Gender-Wise & Grade-Wise Breakdown")
    tbl1, tbl2 = st.columns(2)
    with tbl1:
        st.caption("Gender-Wise Performance Summary")
        st.dataframe(get_gender_performance(df), hide_index=True, width="stretch")
    with tbl2:
        st.caption("Grade Distribution Summary")
        st.dataframe(get_grade_distribution(df), hide_index=True, width="stretch")

    st.markdown("---")
    st.markdown("### Top 10 Performing Students & At-Risk Cohort")
    top_col1, top_col2 = st.columns([1.2, 1.0])
    with top_col1:
        render_figure(plot_top_10_students(df))
    with top_col2:
        st.caption("Top 10 Students Leaderboard")
        top_10 = get_top_students(df, n=10)[
            ["Student_ID", "Name", "Gender", "Attendance_Percentage", "Study_Hours", "Final_Marks", "Grade"]
        ]
        st.dataframe(top_10, hide_index=True, width="stretch")

    with st.expander("View Lowest-Performing Students (Bottom 10)", expanded=False):
        lowest_10 = get_lowest_students(df, n=10)[
            ["Student_ID", "Name", "Gender", "Attendance_Percentage", "Study_Hours", "Final_Marks", "Grade", "Result"]
        ]
        st.dataframe(lowest_10, hide_index=True, width="stretch")

    st.markdown("---")
    st.markdown("### Individual Student Search & Details")
    search_query = st.text_input(
        "Search by Student ID or Student Name",
        placeholder="Example: IT2026001 or Aarav",
    ).strip()

    if search_query:
        mask = (
            df["Student_ID"].str.contains(search_query, case=False, na=False)
            | df["Name"].str.contains(search_query, case=False, na=False)
        )
        matches = df[mask].reset_index(drop=True)
    else:
        matches = df.head(50).reset_index(drop=True)

    if matches.empty:
        st.warning(f"No students matched '{search_query}'. Try another Student ID or Name.")
        return

    options = [f"{row['Student_ID']} — {row['Name']}" for _, row in matches.iterrows()]
    selected_label = st.selectbox("Select Student Record", options=options)
    selected_id = selected_label.split(" — ")[0]
    student = df[df["Student_ID"] == selected_id].iloc[0]
    class_avg = get_overview_metrics(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Final Marks", f"{student['Final_Marks']:.1f}", f"Class Avg: {class_avg['avg_marks']:.1f}")
    with c2:
        render_kpi_card("Grade & Result", f"{student['Grade']} ({student['Result']})", f"Pass Mark: {int(PASS_MARKS)}")
    with c3:
        render_kpi_card(
            "Attendance",
            f"{student['Attendance_Percentage']:.1f}%",
            f"Class Avg: {class_avg['avg_attendance']:.1f}%",
        )
    with c4:
        render_kpi_card(
            "Study Hours",
            f"{student['Study_Hours']:.1f} hrs/day",
            f"Class Avg: {class_avg['avg_study_hours']:.2f} hrs/day",
        )

    st.markdown(
        f"""
        <div class="info-card">
            <strong>Student Academic Profile: {student['Name']} ({student['Student_ID']})</strong><br>
            <span>Gender: <strong>{student['Gender']}</strong> | Age: <strong>{int(student['Age'])}</strong> |
            Previous Marks: <strong>{student['Previous_Marks']:.1f}</strong> |
            Assignment Marks: <strong>{student['Assignment_Marks']:.1f}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_attendance_section(df: pd.DataFrame) -> None:
    """Section C: Attendance distribution, scatter relationship, and insights."""
    st.subheader("C. Attendance Analysis")
    st.write("Analyze class attendance patterns and how attendance impacts final examination marks.")

    att = get_attendance_analysis(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Average Attendance", f"{att['avg_attendance']:.1f}%", f"Median: {att['median_attendance']:.1f}%")
    with c2:
        render_kpi_card("Min / Max Attendance", f"{att['min_attendance']:.1f}% / {att['max_attendance']:.1f}%")
    with c3:
        render_kpi_card("Pass Rate (>= 75%)", f"{att['above_75_pass_rate']:.1f}%", "Good / Excellent attendance")
    with c4:
        render_kpi_card(
            "Pass Rate (< 75%)",
            f"{att['below_75_pass_rate']:.1f}%",
            f"{att['below_75_count']} students below 75%",
        )

    col_left, col_right = st.columns([1.25, 1.0])
    with col_left:
        render_figure(plot_attendance_vs_marks(df))
    with col_right:
        fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor="white")
        sns.histplot(df["Attendance_Percentage"], bins=18, kde=True, color=PRIMARY_BLUE, edgecolor="white", ax=ax)
        ax.axvline(75.0, color="#EF4444", linestyle="--", linewidth=1.8, label="75% Threshold")
        ax.set_title("Attendance Percentage Distribution", fontsize=13, fontweight="bold", color=PRIMARY_NAVY, pad=12)
        ax.set_xlabel("Attendance Percentage (%)", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Number of Students", fontsize=10.5, fontweight="bold")
        ax.legend(frameon=True, facecolor="white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        render_figure(fig)

    st.markdown("### Attendance Brackets & Performance Insights")
    b_col, i_col = st.columns([1.1, 1.0])
    with b_col:
        st.dataframe(att["bracket_summary"], hide_index=True, width="stretch")
    with i_col:
        corr = att["correlation_with_marks"]
        gap = round(att["above_75_pass_rate"] - att["below_75_pass_rate"], 1)
        st.markdown(
            f"""
            <div class="info-card">
                <strong>Attendance-Based Insights</strong>
                <ul style="margin-top:0.5rem; margin-bottom:0;">
                    <li>Correlation between Attendance and Final Marks is <strong>r = {corr:.3f}</strong>.</li>
                    <li>Students maintaining at least 75% attendance outperform peers below 75% by
                    <strong>{gap:+.1f} percentage points</strong> in pass rate.</li>
                    <li><strong>{att['below_75_count']}</strong> students currently fall below the 75% attendance mark
                    and represent a priority group for academic mentoring.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_study_section(df: pd.DataFrame) -> None:
    """Section D: Study hours distribution, scatter plot, and correlation analysis."""
    st.subheader("D. Study Hours & Correlation Analysis")
    st.write("Examine daily study habits and how numeric academic factors correlate with Final Marks.")

    study = get_study_hours_analysis(df)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Avg Study Hours", f"{study['avg_study_hours']:.2f} hrs/day")
    with c2:
        render_kpi_card("Median Study Hours", f"{study['median_study_hours']:.2f} hrs/day")
    with c3:
        render_kpi_card("Study Range", f"{study['min_study_hours']:.1f} - {study['max_study_hours']:.1f} hrs")
    with c4:
        render_kpi_card("Study vs Marks (r)", f"{study['correlation_with_marks']:.3f}", "Pearson correlation")

    col_left, col_right = st.columns([1.25, 1.0])
    with col_left:
        render_figure(plot_study_hours_vs_marks(df))
    with col_right:
        fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor="white")
        sns.histplot(df["Study_Hours"], bins=16, kde=True, color=PRIMARY_NAVY, edgecolor="white", ax=ax)
        ax.axvline(study["avg_study_hours"], color=PRIMARY_BLUE, linestyle="--", linewidth=1.8, label=f"Mean ({study['avg_study_hours']:.2f} hrs)")
        ax.set_title("Daily Study Hours Distribution", fontsize=13, fontweight="bold", color=PRIMARY_NAVY, pad=12)
        ax.set_xlabel("Daily Study Hours", fontsize=10.5, fontweight="bold")
        ax.set_ylabel("Number of Students", fontsize=10.5, fontweight="bold")
        ax.legend(frameon=True, facecolor="white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        render_figure(fig)

    st.markdown("### Study-Hour Brackets & Correlation Matrix")
    st.dataframe(study["bracket_summary"], hide_index=True, width="stretch")

    heat_col, corr_col = st.columns([1.25, 1.0])
    with heat_col:
        render_figure(plot_correlation_heatmap(df))
    with corr_col:
        st.caption("Pearson Correlation Matrix")
        corr_df = get_correlation_matrix(df)
        st.dataframe(corr_df, width="stretch")
        if not corr_df.empty and "Final_Marks" in corr_df.columns:
            final_corrs = corr_df["Final_Marks"].drop(labels=["Final_Marks"], errors="ignore").sort_values(ascending=False)
            top_factor = str(final_corrs.index[0]) if not final_corrs.empty else "N/A"
            top_val = float(final_corrs.iloc[0]) if not final_corrs.empty else 0.0
            st.markdown(
                f"""
                <div class="info-card">
                    <strong>Correlation Summary</strong>
                    <p style="margin:0.4rem 0 0 0;">
                        Among all tracked attributes, <strong>{top_factor}</strong> shows the strongest
                        positive association with Final Marks (<strong>r = {top_val:.3f}</strong>).
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_explorer_section(df: pd.DataFrame) -> None:
    """Section E: Interactive filtering and CSV export."""
    st.subheader("E. Dataset Explorer")
    st.write("Filter the cleaned student dataset by gender, grade, and pass/fail outcome, then export to CSV.")

    f1, f2, f3 = st.columns(3)
    gender_options = sorted(df["Gender"].unique().tolist())
    grade_options = [g for g in GRADE_ORDER if g in df["Grade"].unique()]

    with f1:
        selected_genders = st.multiselect("Filter by Gender", options=gender_options, default=gender_options)
    with f2:
        selected_grades = st.multiselect("Filter by Grade", options=grade_options, default=grade_options)
    with f3:
        selected_result = st.selectbox("Filter by Pass / Fail", options=["All", "Pass", "Fail"], index=0)

    filtered = df.copy()
    if selected_genders:
        filtered = filtered[filtered["Gender"].isin(selected_genders)]
    else:
        filtered = filtered.iloc[0:0]

    if selected_grades:
        filtered = filtered[filtered["Grade"].isin(selected_grades)]
    else:
        filtered = filtered.iloc[0:0]

    if selected_result != "All":
        filtered = filtered[filtered["Result"] == selected_result]

    st.info(f"Showing **{len(filtered)}** of **{len(df)}** student records matching current filters.")

    if filtered.empty:
        st.warning("No records match the selected filters. Adjust one or more filter controls above.")
        return

    st.dataframe(filtered, hide_index=True, width="stretch")

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Filtered Dataset (CSV)",
        data=csv_bytes,
        file_name="filtered_students.csv",
        mime="text/csv",
        type="primary",
    )


def render_report_section(df: pd.DataFrame, cleaning_report: CleaningReport) -> None:
    """Section F: Automated findings, cleaning audit, and TXT report download."""
    st.subheader("F. Automated Analysis Report")
    st.write("Review automatically generated academic findings, data cleaning diagnostics, and download the TXT report.")

    insights = generate_insights(df)
    insights_html = "".join(f"<li>{item}</li>" for item in insights)
    st.markdown(
        f"""
        <div class="info-card">
            <strong>Automated Academic Performance Insights</strong>
            <ol style="margin-top:0.5rem; margin-bottom:0;">
                {insights_html}
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_clean, col_missing = st.columns(2)
    with col_clean:
        st.markdown("#### Data Cleaning & Validation Audit")
        summary_dict = format_cleaning_summary(cleaning_report)
        summary_df = pd.DataFrame(
            [{"Metric": k, "Value": str(v)} for k, v in summary_dict.items()]
        )
        st.dataframe(summary_df, hide_index=True, width="stretch")

    with col_missing:
        st.markdown("#### Missing Values Handled (Before vs After)")
        missing_rows = [
            {
                "Column": col,
                "Missing Before": cleaning_report.missing_before.get(col, 0),
                "Missing After": cleaning_report.missing_after.get(col, 0),
            }
            for col in cleaning_report.missing_before
        ]
        st.dataframe(pd.DataFrame(missing_rows), hide_index=True, width="stretch")

    report_text = generate_text_report(df, cleaning_report=cleaning_report, output_path=REPORT_PATH)

    btn_col, _ = st.columns([1, 2])
    with btn_col:
        st.download_button(
            label="Download Analysis Report (TXT)",
            data=report_text.encode("utf-8"),
            file_name="analysis_report.txt",
            mime="text/plain",
            type="primary",
        )

    with st.expander("Preview Full TXT Report", expanded=True):
        st.code(report_text, language="text")


def main() -> None:
    """Main Streamlit application controller."""
    apply_custom_theme()

    st.markdown(
        """
        <div class="hero-banner">
            <h1>🎓 Student Performance Analysis System</h1>
            <p>B.Tech IT Analytics Dashboard — Data Cleaning, Exploratory Data Analysis &amp; Academic Insights</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not DATA_PATH.exists():
        st.error(f"Dataset file not found at `{DATA_PATH.name}`.")
        st.info("Click the button below to generate the realistic 550-student CSV dataset.")
        if st.button("Generate Student Dataset Now", type="primary"):
            save_dataset(DATA_PATH)
            st.cache_data.clear()
            st.rerun()
        return

    try:
        _, clean_df, cleaning_report = get_cleaned_data(str(DATA_PATH))
    except Exception as exc:
        st.error(f"Failed to load or clean dataset: {exc}")
        return

    if clean_df.empty:
        st.warning("The loaded dataset contains no valid student records.")
        return

    # Sidebar Navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Select Dashboard Section",
        options=[
            "Overview",
            "Student Analysis",
            "Attendance Analysis",
            "Study Analysis",
            "Dataset Explorer",
            "Report",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset Status")
    st.sidebar.write(f"**Clean Records:** {cleaning_report.final_rows}")
    st.sidebar.write(f"**Missing Fixed:** {cleaning_report.total_missing_before}")
    st.sidebar.write(f"**Validation:** {'Passed' if cleaning_report.is_valid else 'Issues Found'}")

    if section == "Overview":
        render_overview_section(clean_df)
    elif section == "Student Analysis":
        render_student_analysis_section(clean_df)
    elif section == "Attendance Analysis":
        render_attendance_section(clean_df)
    elif section == "Study Analysis":
        render_study_section(clean_df)
    elif section == "Dataset Explorer":
        render_explorer_section(clean_df)
    elif section == "Report":
        render_report_section(clean_df, cleaning_report)


if __name__ == "__main__":
    main()
