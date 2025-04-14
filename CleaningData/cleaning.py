# importing packages
import pandas as pd
import os
from datetime import datetime

def extract_bond_info(filename):
    """
    Function that extracts ticker and maturity from the
    filename and returns it as tuple
    """
    name = filename.replace(".xlsx", "")
    parts = name.split("_")

    if len(parts) >= 3:
        rating = parts[0]
        ticker = parts[1]
        maturity = parts[2]
    else:
        rating = "NA"
        ticker = "NA"
        maturity = "NA"

    return rating, ticker, maturity

def clean_bond_file(filepath, filename):
    """ function that takes filepath and filename as inputs
        and returns dictionary with bond information
    """
    raw_data = pd.read_excel(filepath, header=None)
    bond_id = raw_data.iloc[0, 1] if pd.notna(raw_data.iloc[0, 1]) else "NA"
    # reading file and skipping first 5 rows
    df = pd.read_excel(filepath, skiprows=5)
    # dropping empty rows and columns
    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = [str(col).strip() for col in df.columns]

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # getting ticker and maturity using extract_bond_info
    rating, ticker, maturity = extract_bond_info(filename)

    return {
        "data": df,
        "bond_id": bond_id,
        "rating": rating,
        "ticker": ticker,
        "maturity": maturity
    }

def build_bond_dict(base_dir):
    """
    Function that goes through all files and data and 
    returns a nested dictionary with different bonds
    """
    bond_ratings = {}

    for folder in os.listdir(base_dir):
        if folder == "TreasuryData":
            continue
        folder_path = os.path.join(base_dir, folder)
    
        if not os.path.isdir(folder_path):
            continue

        # for loop with error handling
        for filename in os.listdir(folder_path):
            # skipping all non excel files
            if not filename.endswith(".xlsx"):
                continue

            file_path = os.path.join(folder_path, filename)

            try:
                bond_info = clean_bond_file(file_path, filename)
                rating = bond_info["rating"]
                bond_id = bond_info["bond_id"]

                if rating not in bond_ratings:
                    bond_ratings[rating] = {}

                bond_ratings[rating][bond_id] = {
                    "data": bond_info["data"],
                    "ticker": bond_info["ticker"],
                    "maturity": bond_info["maturity"]
                }

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    return bond_ratings
# enter here base path to the folder
base_path = "/Users/maksymnaumenko/Desktop/MF728-RiskAnalysisProject/Data"
bond_ratings = build_bond_dict(base_path)

# How data looks
# for rating in bond_ratings:
#     print(f"\n Rating: {rating}")
#     for bond_id in bond_ratings[rating]:
#         bond = bond_ratings[rating][bond_id]
#         print(f"Bond ID: {bond_id}")
#         print(f"Ticker: {bond['ticker']}")
#         print(f"Maturity: {bond['maturity']}")
#         print(f"Data sample:\n{bond['data'].head()}\n")

for rating in bond_ratings:
    bond_ids = list(bond_ratings[rating].keys())
    print(f"{rating}: {bond_ids}")