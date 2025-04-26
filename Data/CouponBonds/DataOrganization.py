import pandas as pd
import os
import shutil
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

    # I
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

                    # Checking if the file already exists
                    if not os.path.exists(file_path):
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

def combine_files()->None:
    mainwd = Path.cwd()

    for paths in mainwd.iterdir():
        with open_folder(paths):
            folder_path = Path.cwd()

            for csv in folder_path.iterdir():
                print(pd.read_csv(csv))
                break
            break

                
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    # Getting the rating of the bonds from an excel sheet path
    organize_data("Temp_Bloomberg_Data.xlsx", "CouponBondData")
        
    # Cleaning the folders by removing the files that are not needed
    with open_folder("CouponBondData"):
        clean_folders()
        combine_files()





    

