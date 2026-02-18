import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("data/output/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

# load the Excel file
file_path = "data/2026 to 2016 (18 feb).xlsx"
xls = pd.ExcelFile(file_path)

sheet_names = xls.sheet_names

# ensure no more than 1 sheet is loaded
if len(sheet_names) > 1:
    raise ValueError(
        f"Expected only 1 sheet, but found {len(sheet_names)} sheets: {sheet_names}"
    )

# load the single sheet
df = xls.parse(sheet_names[0], header=2)

# ---------------------------------------------------------------------
# Column renaming and standardisation
# ---------------------------------------------------------------------

COLUMN_RENAME_MAP = {
    "UHELDSDATO": "accident_date",
    "UHELDSART": "report_category",
    "KODE_UHELDSSITUATION": "encoded_accident_situation",
    "UHELDSSITUATION": "accident_situation",
    "UHELDSTEKST": "police_narrative",
    "AAR": "year",
}

df = df.rename(columns=COLUMN_RENAME_MAP)

expected_columns = set(COLUMN_RENAME_MAP.values())
missing_columns = expected_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

df = df[list(COLUMN_RENAME_MAP.values())]


# ---------------------------------------------------------------------
# Basic validation and helper columns
# ---------------------------------------------------------------------

df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["has_narrative"] = df["police_narrative"].notna()

if df["year"].isna().any():
    raise ValueError("Year column contains non-numeric values")


# ---------------------------------------------------------------------
# Dataset size and temporal coverage
# ---------------------------------------------------------------------

total_reports = len(df)

reports_per_year = (
    df.groupby("year")
      .size()
      .reset_index(name="n_reports")
      .sort_values("year")
)

print(f"Total number of reports: {total_reports}")
print(reports_per_year)

#plotting the number of reports per year
plt.figure()
plt.bar(reports_per_year["year"], reports_per_year["n_reports"])
plt.xlabel("Year")
plt.ylabel("Number of reports")
plt.title("Number of accident reports per year")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "reports_per_year.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# Accident classification distribution
# ---------------------------------------------------------------------

classification_dist = (
    df["report_category"]
      .value_counts(dropna=False)
      .reset_index()
)

classification_dist.columns = ["report_category", "count"]
classification_dist["share"] = (
    classification_dist["count"] / classification_dist["count"].sum()
)

print(classification_dist)

#plotting the distribution of accident classifications
plt.figure()
plt.bar(classification_dist["report_category"],
        classification_dist["count"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Number of reports")
plt.title("Distribution of accident classifications")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accident_classification_distribution.png", dpi=300)
plt.close()
