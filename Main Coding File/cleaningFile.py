import pandas as pd
import re
import os
from pathlib import Path

def parse_bond_sheet(sheet_df):
    """exxtracting bond data and timeseries data"""
    cusip = sheet_df.iloc[3, 1]
    company = sheet_df.iloc[1, 1]
    ticker = sheet_df.iloc[2, 1]
    rating = sheet_df.iloc[4, 1]
    issue_date = pd.to_datetime(sheet_df.iloc[5, 1], errors='coerce').date()
    maturity_date = pd.to_datetime(sheet_df.iloc[6, 1], errors='coerce').date()
    coupon = pd.to_numeric(sheet_df.iloc[7, 1], errors='coerce')

    ts_data = sheet_df.iloc[9:].copy()
    ts_data.columns = sheet_df.iloc[9]
    ts_data = ts_data.iloc[1:]
    ts_data.dropna(how='all', inplace=True)
    ts_data.dropna(axis=1, how='all', inplace=True)

    ts_data.columns = [str(c).strip() for c in ts_data.columns]
    if 'Date' in ts_data.columns:
        ts_data['Date'] = pd.to_datetime(ts_data['Date'], errors='coerce')
    if 'Mid Price' in ts_data.columns:
        ts_data['Mid Price'] = pd.to_numeric(ts_data['Mid Price'], errors='coerce')
    if 'Mid YTM' in ts_data.columns:
        ts_data['Mid YTM'] = pd.to_numeric(ts_data['Mid YTM'], errors='coerce')

    ts_data.reset_index(drop=True, inplace=True) 

    return cusip, {
        "company": company,
        "ticker": ticker,
        "issue_date": issue_date,
        "maturity_date": maturity_date,
        "coupon": coupon,
        "data": ts_data
    }

def normalize_rating(sheet_name):
    """cleaning sheet names"""
    return re.sub(r"\s*\(\d+\)$", "", sheet_name.strip())

def build_bond_dict_from_excel(file_path):
    """Reads multi-tab Excel file and creates nested bond dictionary with normalized ratings"""
    xls = pd.ExcelFile(file_path)
    bond_dict = {}

    for sheet_name in xls.sheet_names:
        clean_name = sheet_name.strip().lower()
        if clean_name in {"temp", "data"}:
            continue

        sheet_df = xls.parse(sheet_name, header=None)
        cusip, bond_info = parse_bond_sheet(sheet_df)
        rating = normalize_rating(sheet_name)

        if rating not in bond_dict:
            bond_dict[rating] = {}
        bond_dict[rating][cusip] = bond_info

    return bond_dict

# enter here base path to the folder
file_path = os.getcwd() + "Data\CouponBonds\Temp_Bloomberg_Data_Formulas.xlsx" 
bond_data = build_bond_dict_from_excel(file_path)

# See structure of the data
for rating in bond_data:
    print(f"\nRating: {rating}")
    for cusip in bond_data[rating]:
        bond = bond_data[rating][cusip]
        print(f"Cusip: {cusip}")
        print(f"Company: {bond['company']}")
        print(f"Ticker: {bond['ticker']}")
        print(f"Issue Date: {bond['issue_date']}")
        print(f"Maturity: {bond['maturity_date']}")
        print(f"Coupon: {bond['coupon']}")
        print(f"Data sample:\n{bond['data'].head()}\n")

# # see all ratings and bonds IDs
# for rating in bond_data:
#     print(f"\nRating: {rating}")
#     for cusip in bond_data[rating]:
#         print(f"Cusip: {cusip}")
