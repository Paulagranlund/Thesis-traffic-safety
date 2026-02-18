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


# ---------------------------------------------------------------------
# Accident classifications over time
# ---------------------------------------------------------------------

class_year = (
    df.groupby(["year", "report_category"])
      .size()
      .reset_index(name="count")
)

class_year["share"] = (
    class_year["count"] /
    class_year.groupby("year")["count"].transform("sum")
)

pivot_class_year = class_year.pivot(
    index="year",
    columns="report_category",
    values="share"
)

#plotting the shares of accident classifications over time
# Ensure year order
pivot_class_year = pivot_class_year.sort_index()

plt.figure()
for col in pivot_class_year.columns:
    plt.plot(
        pivot_class_year.index.to_numpy(),
        pivot_class_year[col].to_numpy(),
        label=str(col)
    )

plt.xlabel("Year")
plt.ylabel("Share of reports")
plt.title("Accident classification shares over time")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accident_classification_over_time_lines.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------
# Distribution of main accident situations
# ---------------------------------------------------------------------

df["encoded_accident_situation"] = pd.to_numeric(
    df["encoded_accident_situation"], errors="coerce"
)

# Main class: 0 if <100, 1 if 100-199, 2 if 200-299, ...
df["main_situation_class"] = (df["encoded_accident_situation"] // 100).astype("Int64")


main_class_dist = (
    df["main_situation_class"]
      .value_counts(dropna=False)
      .sort_index()
      .reset_index()
)
main_class_dist.columns = ["main_situation_class", "count"]
main_class_dist["share"] = main_class_dist["count"] / main_class_dist["count"].sum()
print("Non-numeric situation codes:", df["encoded_accident_situation"].isna().sum())

print(main_class_dist)


plt.figure()
plt.bar(
    main_class_dist["main_situation_class"].astype(str).to_numpy(),
    main_class_dist["count"].to_numpy()
)
plt.xlabel("Main situation class (hundreds bucket)")
plt.ylabel("Number of reports")
plt.title("Distribution of main accident situation classes")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "main_situation_class_distribution.png", dpi=300)
plt.close()

situation_codebook = (
    df.dropna(subset=["encoded_accident_situation"])
      .groupby("encoded_accident_situation")["accident_situation"]
      .agg(lambda s: s.dropna().astype(str).value_counts().index[0] if len(s.dropna()) else "")
      .reset_index()
      .rename(columns={"accident_situation": "situation_name"})
)

sub_by_class = (
    df.dropna(subset=["main_situation_class", "encoded_accident_situation"])
      .groupby(["main_situation_class", "encoded_accident_situation"])
      .size()
      .reset_index(name="count")
)

sub_by_class["share_within_class"] = (
    sub_by_class["count"] /
    sub_by_class.groupby("main_situation_class")["count"].transform("sum")
)

sub_by_class = sub_by_class.merge(
    situation_codebook, on="encoded_accident_situation", how="left"
)

sub_by_class = sub_by_class.sort_values(
    ["main_situation_class", "count"], ascending=[True, False]
)

print(sub_by_class.head(30))

#TOP_N = 10

#top_sub_by_class = (
#    sub_by_class
#    .groupby("main_situation_class", group_keys=False)
#    .head(TOP_N)
#)

#print(top_sub_by_class)


sub_by_class.to_csv("data/output/sub_situations_by_class_full.csv", index=False)
#top_sub_by_class.to_csv("data/output/sub_situations_by_class_top10.csv", index=False)

for cls, grp in sub_by_class.groupby("main_situation_class"):
    labels = (
        grp["encoded_accident_situation"].astype(int).astype(str)
        + " - "
        + grp["situation_name"].fillna("")
    )

    plt.figure()
    plt.bar(labels.to_numpy(), grp["share_within_class"].to_numpy())
    plt.xticks(rotation=90, ha="right")
    plt.xlabel("Sub accident situation (code + name)")
    plt.ylabel("Share within main class")
    plt.title(f"Sub accident situations within class {cls}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"sub_situations_top_class_{cls}.png", dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Distribution of main situation classes by accident classification
# ---------------------------------------------------------------------

base_counts = (
    df.dropna(subset=["main_situation_class", "report_category"])
      .groupby(["main_situation_class", "report_category"])
      .size()
      .reset_index(name="count")
)


base_counts["share_within_classification"] = (
    base_counts["count"] /
    base_counts.groupby("report_category")["count"].transform("sum")
)

base_counts["share_within_main_situation"] = (
    base_counts["count"] /
    base_counts.groupby("main_situation_class")["count"].transform("sum")
)
pivot_A = base_counts.pivot(
    index="report_category",
    columns="main_situation_class",
    values="share_within_classification"
).fillna(0)

plt.figure(figsize=(10, 6))
pivot_A.plot(kind="bar", stacked=True, width=0.8)
plt.xlabel("Accident classification")
plt.ylabel("Share within classification")
plt.title("Main situation class distribution within each accident classification")
plt.legend(title="Main situation class", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "main_situation_within_classification_stacked.png", dpi=300)
plt.close()
pivot_B = base_counts.pivot(
    index="main_situation_class",
    columns="report_category",
    values="share_within_main_situation"
).fillna(0).sort_index()

plt.figure(figsize=(10, 6))
pivot_B.plot(kind="bar", stacked=True, width=0.8)
plt.xlabel("Main situation class")
plt.ylabel("Share within main situation class")
plt.title("Accident classification distribution within each main situation class")
plt.legend(title="Accident classification", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "classification_within_main_situation_stacked.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# Availability of VD narratives
# ---------------------------------------------------------------------

overall_narrative_share = df["has_narrative"].mean()
print(f"Overall share with narrative: {overall_narrative_share:.2%}")

narrative_by_year = (
    df.groupby("year")["has_narrative"]
      .mean()
      .reset_index(name="share_with_narrative")
)

narrative_by_class = (
    df.groupby("report_category")["has_narrative"]
      .mean()
      .reset_index(name="share_with_narrative")
)

#plotting the VD narrative availability by year
plt.figure()
plt.plot(
    narrative_by_year["year"].to_numpy(),
    narrative_by_year["share_with_narrative"].to_numpy()
)
plt.xlabel("Year")
plt.ylabel("Share with narrative")
plt.title("Narrative availability by year")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "narrative_availability_by_year.png", dpi=300)
plt.close()

no_narrative_by_year = (
    df.loc[~df["has_narrative"]]
      .groupby("year")
      .size()
      .reset_index(name="n_without_narrative")
      .sort_values("year")
)

print(no_narrative_by_year)


# ---------------------------------------------------------------------
# Freeze full dataset before text-based filtering
# ---------------------------------------------------------------------

df_full = df.copy()

# ---------------------------------------------------------------------
# Subset for VD narrative analysis
# ---------------------------------------------------------------------

df_text = df_full.loc[df_full["has_narrative"]].copy()




# ---------------------------------------------------------------------
# Narrative length measures
# ---------------------------------------------------------------------

df_text["n_chars"] = df_text["police_narrative"].str.len()
df_text["n_words"] = df_text["police_narrative"].str.split().str.len()
df_text["n_sentences"] = (
    df_text["police_narrative"]
    .str.count(r"[.!?]+")
)


length_summary = df_text[["n_chars", "n_words", "n_sentences"]].describe(
    percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
)

print(length_summary)


plt.figure()
plt.hist(df_text["n_words"].to_numpy(), bins=50)
plt.xlabel("Number of words")
plt.ylabel("Frequency")
plt.title("Distribution of VD narrative length (words)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vd_narrative_word_length_distribution.png", dpi=300)
plt.close()


plt.figure()
plt.hist(df_text["n_sentences"].to_numpy(), bins=40)
plt.xlabel("Number of sentences")
plt.ylabel("Frequency")
plt.title("Distribution of VD narrative length (sentences)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "vd_narrative_sentence_length_distribution.png", dpi=300)
plt.close()

# ---------------------------------------------------------------------
# Export VD narratives with fewer than 3 words
# ---------------------------------------------------------------------

mask = df_text["n_words"] < 3

cols_to_save = [
    "police_narrative",
    "n_words",
    "year",
    "report_category",
    "main_situation_class",
]

short_narratives = df_text.loc[mask, cols_to_save]

output_path = Path("data/output/short_vd_narratives_less_than_3_words.csv")
short_narratives.to_csv(output_path, index=False)

print(
    f"Saved {len(short_narratives)} VD narratives with fewer than 3 words to {output_path}"
)
