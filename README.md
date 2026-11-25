# Student Performance Analysis System

## 1. Project Introduction
The **Student Performance Analysis System** is a B.Tech IT mini project built with Python, Pandas, NumPy, Matplotlib, Seaborn, and Streamlit. It provides an end-to-end data analytics pipeline that generates a realistic student dataset, cleans and validates the data without modifying the original CSV file, performs Exploratory Data Analysis (EDA), renders visual charts, and presents an interactive web dashboard along with a downloadable summary report.

---

## 2. Problem Statement
Educational institutions collect large amounts of academic data such as attendance percentages, assignment scores, daily study hours, and examination marks. However, raw records often contain missing or inconsistent values, and manual spreadsheet analysis makes it difficult to spot patterns, identify at-risk students early, or understand which factors most strongly influence final exam performance.

---

## 3. Project Objectives
- Generate a reproducible synthetic dataset of at least 500 B.Tech IT student records (`students.csv`).
- Demonstrate real-world data cleaning (missing value handling, duplicate removal, type correction, range validation, and grade recalculation) without altering the original raw CSV file.
- Perform comprehensive Exploratory Data Analysis (EDA) across marks, attendance, study hours, gender, and grades.
- Visualize academic trends using clean, informative Matplotlib and Seaborn charts.
- Deliver a responsive, blue-and-white Streamlit dashboard with search, filters, CSV export, and automated TXT report generation.

---

## 4. Features
- **Reproducible Dataset Generator (`data_generator.py`)**: Generates 550 realistic student records using a fixed random seed (`42`) and controlled missing values for cleaning demonstration.
- **Automated Data Cleaning (`data_cleaning.py`)**: Detects and imputes missing values, removes duplicates, fixes invalid values/data types, validates schema and value ranges, and verifies consistent grades (`A+`, `A`, `B`, `C`, `D`, `F`) and pass/fail status (`Pass >= 40`).
- **Exploratory Data Analysis (`analysis.py`)**: Computes KPI metrics, gender-wise performance, grade and performance tier distributions, top 10 & lowest-performing students, attendance/study bracket summaries, correlation matrix, and automated academic conclusions.
- **Data Visualizations (`visualization.py`)**: Generates 8 styled charts (Marks Histogram, Grade Bar Chart, Gender Average Marks, Attendance vs Marks Scatter, Study Hours vs Marks Scatter, Top 10 Students Bar Chart, Pass vs Fail Pie Chart, and Correlation Heatmap).
- **Interactive Streamlit Dashboard (`app.py`)**:
  - **Overview**: KPI cards, marks distribution, pass/fail chart, and performance summary.
  - **Student Analysis**: Gender-wise stats, grade breakdown, top 10 students, and individual student search.
  - **Attendance Analysis**: Attendance distribution, attendance vs marks plot, and 75% attendance pass-rate comparison.
  - **Study Analysis**: Study hours distribution, scatter plot, and correlation heatmap.
  - **Dataset Explorer**: Multi-filter explorer (Gender, Grade, Pass/Fail) and filtered CSV download.
  - **Report**: Automated insights, cleaning audit table, and downloadable `analysis_report.txt`.

---

## 5. Technology Used
- **Python 3.13**
- **Pandas** (Data manipulation, cleaning, grouping, CSV I/O)
- **NumPy** (Reproducible numerical data generation)
- **Matplotlib & Seaborn** (Statistical charts and heatmaps)
- **Streamlit** (Interactive web dashboard)
- **CSV** (`students.csv` flat-file dataset)

---

## 6. Project Folder Structure
```text
student_performance_analysis/
│
├── app.py                      # Streamlit dashboard application
├── analysis.py                 # EDA computations and TXT report generator
├── data_generator.py           # Reproducible synthetic dataset generator
├── data_cleaning.py            # Cleaning, validation, and grade processing
├── visualization.py            # 8 Matplotlib & Seaborn visualization functions
├── students.csv                # Generated student dataset (550 records)
├── requirements.txt            # Python package dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
│
└── reports/
    └── analysis_report.txt     # Generated academic analysis report
```

---

## 7. Installation Instructions
1. Open a terminal and navigate to the project root folder:
   ```bash
   cd student_performance_analysis
   ```
2. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## 8. Execution Commands
Run the following commands from the `student_performance_analysis/` project root directory:

1. Generate the dataset (`students.csv`):
   ```bash
   python data_generator.py
   ```
2. Launch the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```
3. Open the local URL shown in the terminal (usually `http://localhost:8501`) in your web browser.

---

## 9. Dataset Description
The dataset (`students.csv`) contains **550 student records** across **11 columns**:

| Column | Description |
| :--- | :--- |
| `Student_ID` | Unique identifier (e.g., `IT2026001`) |
| `Name` | Student full name |
| `Gender` | `Male`, `Female`, or `Other` |
| `Age` | Student age in years (18–24) |
| `Attendance_Percentage` | Class attendance percentage (0–100) |
| `Study_Hours` | Daily self-study hours |
| `Previous_Marks` | Previous semester marks (0–100) |
| `Assignment_Marks` | Continuous assignment marks (0–100) |
| `Final_Marks` | Final examination marks (0–100) |
| `Grade` | Letter grade (`A+`: 90+, `A`: 80–89.9, `B`: 70–79.9, `C`: 60–69.9, `D`: 40–59.9, `F`: <40) |
| `Result` | `Pass` (Final Marks >= 40) or `Fail` (Final Marks < 40) |

---

## 10. Expected Output
- Running `python data_generator.py` creates `students.csv` with 550 records and prints dataset summary statistics.
- Running `streamlit run app.py` launches an interactive blue-and-white web dashboard displaying KPI cards, 8 analytical charts, student search, dataset filters with CSV export, and a downloadable `analysis_report.txt` file saved under `reports/`.

---

## 11. Conclusion
The Student Performance Analysis System demonstrates how Python data analytics tools can convert raw student records into actionable academic insights. Through automated data cleaning, correlation analysis, and interactive visualization, faculty and students can easily monitor attendance impact, study habits, and overall batch performance.

---

## 12. Future Scope
- Support uploading custom CSV/Excel files directly from the dashboard.
- Add machine learning models (e.g., Linear Regression or Random Forest) to predict at-risk students early in the semester.
- Integrate semester-over-semester longitudinal tracking and PDF report export.
- Connect to a SQL database for multi-department college deployments.
