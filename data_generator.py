"""Dataset generator for the Student Performance Analysis System.

Generates a reproducible, realistic synthetic CSV dataset of B.Tech IT student
performance records including controlled missing values for data-cleaning
demonstration.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED: int = 42
DEFAULT_NUM_RECORDS: int = 550
PASS_MARKS: float = 40.0
OUTPUT_CSV_PATH: Path = Path(__file__).resolve().parent / "students.csv"

FIRST_NAMES_MALE: tuple[str, ...] = (
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan",
    "Krishna", "Ishaan", "Shaurya", "Atharv", "Pranav", "Kabir", "Rohan",
    "Karthik", "Siddharth", "Rahul", "Nikhil", "Varun", "Harsh", "Dev",
    "Yash", "Tanmay", "Ankit", "Gaurav", "Manish", "Abhishek", "Kunal", "Raj",
)

FIRST_NAMES_FEMALE: tuple[str, ...] = (
    "Aadhya", "Ananya", "Diya", "Saanvi", "Pari", "Anika", "Navya", "Myra",
    "Aarohi", "Ira", "Kavya", "Riya", "Sneha", "Pooja", "Neha", "Priya",
    "Shreya", "Ishita", "Tanvi", "Meera", "Nandini", "Kritika", "Aditi",
    "Simran", "Muskan", "Khushi", "Divya", "Sakshi", "Bhavna", "Swati",
)

FIRST_NAMES_NEUTRAL: tuple[str, ...] = (
    "Aryan", "Kiran", "Sam", "Surya", "Tejas", "Harpreet", "Robin", "Jatin",
)

LAST_NAMES: tuple[str, ...] = (
    "Sharma", "Verma", "Patel", "Gupta", "Singh", "Kumar", "Reddy", "Nair",
    "Iyer", "Joshi", "Kulkarni", "Mehta", "Malhotra", "Bansal", "Agarwal",
    "Chauhan", "Rao", "Desai", "Pandey", "Mishra", "Tiwari", "Saxena",
    "Kapoor", "Bhat", "Das", "Bose", "Sen", "Chatterjee", "Menon", "Pillai",
)


def calculate_grade(marks: float) -> str:
    """Return letter grade for a numeric mark on a 0-100 scale."""
    if pd.isna(marks):
        return "F"
    if marks >= 90:
        return "A+"
    if marks >= 80:
        return "A"
    if marks >= 70:
        return "B"
    if marks >= 60:
        return "C"
    if marks >= PASS_MARKS:
        return "D"
    return "F"


def calculate_result(marks: float) -> str:
    """Return 'Pass' if marks >= PASS_MARKS else 'Fail'."""
    if pd.isna(marks):
        return "Fail"
    return "Pass" if marks >= PASS_MARKS else "Fail"


def generate_student_dataset(
    num_records: int = DEFAULT_NUM_RECORDS,
    seed: int = RANDOM_SEED,
    inject_missing: bool = True,
) -> pd.DataFrame:
    """Generate a realistic student performance DataFrame.

    Parameters
    ----------
    num_records:
        Number of unique student records to generate (minimum recommended: 500).
    seed:
        Fixed random seed for full reproducibility.
    inject_missing:
        Whether to inject controlled missing values (~3%) for cleaning demo.

    Returns
    -------
    pd.DataFrame
        Generated student performance dataset.
    """
    if num_records < 1:
        raise ValueError("num_records must be at least 1.")

    rng = np.random.default_rng(seed)

    # Unique, non-duplicated Student IDs
    student_ids = [f"IT2026{idx:03d}" for idx in range(1, num_records + 1)]

    # Gender distribution
    genders = rng.choice(
        ["Male", "Female", "Other"],
        size=num_records,
        p=[0.51, 0.46, 0.03],
    )

    # Generate realistic names matched to gender
    names: list[str] = []
    for gender in genders:
        if gender == "Male":
            first = str(rng.choice(FIRST_NAMES_MALE))
        elif gender == "Female":
            first = str(rng.choice(FIRST_NAMES_FEMALE))
        else:
            first = str(rng.choice(FIRST_NAMES_NEUTRAL))
        last = str(rng.choice(LAST_NAMES))
        names.append(f"{first} {last}")

    # B.Tech student ages (18-24, centered around 20-21)
    ages = rng.choice(
        [18, 19, 20, 21, 22, 23, 24],
        size=num_records,
        p=[0.06, 0.16, 0.26, 0.27, 0.16, 0.06, 0.03],
    ).astype(float)

    # Underlying latent academic engagement factor (controls realistic correlations)
    engagement = rng.normal(loc=0.0, scale=1.0, size=num_records)

    # Attendance percentage (mostly 55-98%, correlated with engagement)
    attendance = np.clip(78.0 + 11.0 * engagement + rng.normal(0, 6.5, num_records), 32.0, 100.0)
    attendance = np.round(attendance, 1)

    # Daily study hours (0.5 to 8.0 hrs/day, correlated with engagement)
    study_hours = np.clip(3.2 + 1.3 * engagement + rng.normal(0, 0.9, num_records), 0.5, 8.5)
    study_hours = np.round(study_hours, 1)

    # Previous semester marks (correlated with engagement)
    previous_marks = np.clip(64.0 + 12.0 * engagement + rng.normal(0, 8.0, num_records), 22.0, 99.0)
    previous_marks = np.round(previous_marks, 1)

    # Assignment marks (generally slightly higher than exam marks)
    assignment_marks = np.clip(
        0.55 * previous_marks + 0.35 * attendance + 10.0 + rng.normal(0, 6.0, num_records),
        25.0,
        100.0,
    )
    assignment_marks = np.round(assignment_marks, 1)

    # Final marks driven realistically by study hours, attendance, previous & assignment marks
    raw_final = (
        0.32 * previous_marks
        + 0.28 * assignment_marks
        + 0.22 * attendance
        + 3.2 * study_hours
        - 4.5
        + rng.normal(0, 5.5, num_records)
    )
    final_marks = np.round(np.clip(raw_final, 12.0, 99.5), 1)

    grades = [calculate_grade(val) for val in final_marks]
    results = [calculate_result(val) for val in final_marks]

    df = pd.DataFrame(
        {
            "Student_ID": student_ids,
            "Name": names,
            "Gender": genders,
            "Age": ages,
            "Attendance_Percentage": attendance,
            "Study_Hours": study_hours,
            "Previous_Marks": previous_marks,
            "Assignment_Marks": assignment_marks,
            "Final_Marks": final_marks,
            "Grade": grades,
            "Result": results,
        }
    )

    if inject_missing:
        # Inject controlled missing values (~2-3% per selected column) for cleaning demo
        missing_specs: dict[str, int] = {
            "Gender": max(3, int(0.015 * num_records)),
            "Age": max(5, int(0.02 * num_records)),
            "Attendance_Percentage": max(8, int(0.03 * num_records)),
            "Study_Hours": max(8, int(0.03 * num_records)),
            "Previous_Marks": max(6, int(0.025 * num_records)),
            "Assignment_Marks": max(6, int(0.025 * num_records)),
            "Final_Marks": max(4, int(0.015 * num_records)),
        }
        for col, count in missing_specs.items():
            chosen_idx = rng.choice(num_records, size=count, replace=False)
            df.loc[chosen_idx, col] = np.nan

        # Also blank out a few Grade/Result cells where Final_Marks is missing or valid
        grade_idx = rng.choice(num_records, size=max(4, int(0.015 * num_records)), replace=False)
        df.loc[grade_idx, "Grade"] = np.nan
        df.loc[grade_idx, "Result"] = np.nan

    # Store Age cleanly as nullable Int64 when missing values exist
    df["Age"] = df["Age"].astype("Int64")
    return df


def save_dataset(
    output_path: Path | str = OUTPUT_CSV_PATH,
    num_records: int = DEFAULT_NUM_RECORDS,
    seed: int = RANDOM_SEED,
) -> Path:
    """Generate the dataset and save it to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_student_dataset(num_records=num_records, seed=seed, inject_missing=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    """Entry point to generate students.csv in the project root."""
    output_path = save_dataset()
    df = pd.read_csv(output_path)
    total_missing = int(df.isna().sum().sum())
    print(f"Dataset generated successfully at: {output_path}")
    print(f"Total records: {len(df)} | Unique Student_IDs: {df['Student_ID'].nunique()}")
    print(f"Total injected missing cells (for cleaning demo): {total_missing}")


if __name__ == "__main__":
    main()
