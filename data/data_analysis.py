import pandas as pd


#load the Excel file
file_path ="data/2026 to 2016 (18 feb).xlsx"
xls = pd.ExcelFile(file_path)
    
table_name = file_path.split("/")[-1]  
    
sheet_names = xls.sheet_names

# ensure no more than 1 sheet is loaded
if len(sheet_names) > 1:
    raise ValueError(f"Expected only 1 sheet, but found {len(sheet_names)} sheets: {sheet_names}")

# load the single sheet
df = xls.parse(sheet_names[0], nrows=0, header=2)

# print number of rows and columns
print(df)
num_rows, num_cols = df.shape
print(f"Number of rows: {num_rows}")
print(f"Number of columns: {num_cols}")

#print

