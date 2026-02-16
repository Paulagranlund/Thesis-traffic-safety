
import pandas as pd

def extract_excel_info(file_path):

    xls = pd.ExcelFile(file_path)
    
    table_name = file_path.split("/")[-1]  
    
    sheet_names = xls.sheet_names

    sheet_headers = {}
    for sheet in sheet_names:
        df = xls.parse(sheet, nrows=0, header=2) 
        sheet_headers[sheet] = df.columns.tolist()
    
    return {
        "Table Name": table_name,
        "Number of Sheets": len(sheet_names),
        "Sheet Names": sheet_names,
        "Sheet Headers": sheet_headers
    }

def extract_text_and_labels(file_path, sheet_number):
    """
    Returns the columns 'KODE_UHELDSSITUATION' and 'UHELDSTEKST'
    from a given Excel sheet.
    """

    df = pd.read_excel(
        file_path,
        sheet_name=sheet_number,
        header=2, 
        nrows=10000
    )

    required_cols = ['KODE_UHELDSSITUATION', 'UHELDSTEKST']

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in sheet {sheet_number}: {missing}")

    #### MUTED FOR TESTING
    #df['MANCOLL'] = (
    #df['KODE_UHELDSSITUATION']
    #.astype('category')
    #.cat.codes + 1)

    df['SUMMARY'] = df['UHELDSTEKST']

    ##### FOR TESTING SHOULD BE REMOVED LATER

    # Create artificial grouped labels
    df['MANCOLL'] = (
        df['KODE_UHELDSSITUATION'] // 100
    )

    return_cols = ['SUMMARY', 'MANCOLL']
    return df[return_cols]


file_path = "Our Analysis/data/2025 only.xlsx"  
excel_info = extract_excel_info(file_path)

print(f"Table Name: {excel_info['Table Name']}")
print(f"Number of Sheets: {excel_info['Number of Sheets']}")
print("Sheet Names and Headers:")

for sheet, headers in excel_info["Sheet Headers"].items():
    print(f"  - {sheet}: {headers}")



data = extract_text_and_labels("Our Analysis/data/2025 only.xlsx", sheet_number=0)

print(data)
