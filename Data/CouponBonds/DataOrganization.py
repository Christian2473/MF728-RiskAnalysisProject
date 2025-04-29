import pandas as pd
import os
import shutil
import datetime as dt
from pathlib import Path
from contextlib import contextmanager
from typing import List

@contextmanager
def open_folder(path: str | Path):
    """Context Manager that opens the folder and getting the path

    Args:
        path (str): Path to the folder
    """
    try:
        cwd = os.getcwd()
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)


def organize_data(path: str|Path, folder_path: None) -> None:
    """Getting the rating of the bonds from an excel sheet path

    Args:
        path (str): path to the excel file
        folder_path (str): a folder to put all organized data
    """

    excel_file: pd.ExcelFile = pd.ExcelFile(path)
    sheets: List[str] = excel_file.sheet_names

    os.makedirs("CouponBondData", exist_ok=True)

    if not folder_path:
        os.makedirs(folder_path, exist_ok=True) 

    with open_folder(folder_path):
        for sheet_name in sheets:

            if sheet_name in ["Data", "Temp", "", " "]:
                continue
            else:
                rating: str = sheet_name.split(" ")[0]
                if rating == "NR":
                    continue

                os.makedirs(rating, exist_ok=True)

                # Changing the Directory
                with open_folder(rating):
                    file_path = os.path.join(os.getcwd(), sheet_name + ".csv")

                    sheet_df = excel_file.parse(sheet_name)
                    sheet_df.to_csv(f"{sheet_name}.csv", index=False)

def clean_folders(num_files: int = 4) -> None:
    """Cleaning the folders by removing the files that are not needed
    
    Args:
        num_files (int, optional): Number of files threshold to keep. Defaults to 4.
    """

    for folder in os.listdir():
        if os.path.isdir(folder):

            boolean = False #pointer 
            with open_folder(folder):
                files = os.listdir()

                boolean = len(files) < num_files

            if boolean:
                shutil.rmtree(folder)

def combine_files() -> None:
    mainwd = Path.cwd()

    os.makedirs("YieldData", exist_ok=True)
    os.makedirs("PriceData", exist_ok=True)


    for paths in mainwd.iterdir():
        with open_folder(paths):
            folder_path = Path.cwd()

            price_df = pd.DataFrame()
            yield_df = pd.DataFrame()

            if folder_path.name == "PriceData" or folder_path.name == "YieldData":
                continue


            for i, csv in enumerate(folder_path.iterdir()):
                
                df = pd.read_csv(csv)

                #creating the info dataframe
                df_info = df.head(8).dropna(axis=1, how='all').dropna(axis=0, how='all')  # Drop rows and columns that are completely empty
                df_info = df_info.transpose().reset_index(drop=True) # Set the first column as index
                df_info.columns = df_info.iloc[0]  # Set the first row as column names
                df_info = df_info.drop(df_info.index[0])  # Drop the first row which is now redundant   
                df_info = df_info.squeeze()  # Convert the DataFrame to a Series
                df_info.index.name = None
                df_info.name = df_info["Company"]
                
                df_info["Maturity Date"] = dt.datetime.strptime(df_info["Maturity Date"].split(" ")[0], "%m/%d/%Y")  # Remove spaces from the "Issue Date" column
                df_info["Issue Date"] = dt.datetime.strptime(df_info["Issue Date"].split(" ")[0], "%m/%d/%Y")  # Remove spaces from the "Issue Date" column
                df_info["Tenor (days)"] = (df_info["Maturity Date"] - df_info["Issue Date"]).days

                # Creating the price and yield dataframe
                df_price_yield = df.tail(-8)  # Drop columns that are completely empty
                df_price_yield.columns = list(df_price_yield.iloc[0]) # Set the first row as column names
                df_price_yield = df_price_yield.reset_index(drop=True)
                df_price_yield.drop(0, inplace=True)  # Drop the first row which is now redundant
                df_price_yield.set_index('Date', inplace=True)  # Set the first column as index
                df_price_yield = df_price_yield.sort_index(ascending=True)  # Sort the index in ascending order

                #setting up the two dataframes
                yield_df_temp = pd.concat([df_info, df_price_yield["Mid YTM"]], ignore_index=False)
                price_df_temp = pd.concat([df_info, df_price_yield["Mid Price"]], ignore_index=False)

                #naming the columns
                yield_df_temp.name = yield_df_temp["CUSIP"]
                price_df_temp.name = price_df_temp["CUSIP"]

                # Changing to dataframes
                yield_df_temp = yield_df_temp.to_frame()
                price_df_temp = price_df_temp.to_frame()

                # Concatenate the dataframes
                if i == 0:
                    price_df = price_df_temp
                    yield_df = yield_df_temp
                else:
                    price_df = price_df.join(price_df_temp, how='inner', rsuffix=f"_{i}")
                    
                    yield_df = yield_df.join(yield_df_temp, how='inner', rsuffix=f"_{i}")

            # Dropping the CUSIP column
            price_df.drop("CUSIP", axis=0, inplace=True)
            yield_df.drop("CUSIP", axis=0, inplace=True)

            price_df = price_df.dropna(axis=1, how='all')  # Drop columns that are completely empty
            yield_df = yield_df.dropna(axis=1, how='all')  # Drop columns that are completely empty


            # Saving the dataframes
            with open_folder(folder_path.parent / "PriceData"):
                price_df.to_csv(f"{folder_path.name} Price.csv", index=True)
            
            with open_folder(folder_path.parent / "YieldData"):
                yield_df.to_csv(f"{folder_path.name} Yield.csv", index = True)

                
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    # Getting the rating of the bonds from an excel sheet path
    organize_data("Temp_Bloomberg_Data.xlsx", "CouponBondData")
        
    # Cleaning the folders by removing the files that are not needed
    with open_folder("CouponBondData"):
        clean_folders()
        combine_files()





    

