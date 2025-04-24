import pandas as pd
import os
import shutil
from contextlib import contextmanager
from typing import List

@contextmanager
def open_folder(path: str):
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


def organize_data(path: str) -> None:
    """Getting the rating of the bonds from an excel sheet path

    Args:
        path (str): Path to the excel file
    """
    excel_file: pd.ExcelFile = pd.ExcelFile(path)
    sheets: List[str] = excel_file.sheet_names
    sheets: list[str] = excel_file.sheet_names

    try:
        os.makedirs("CouponBondData")
    except FileExistsError:
        pass
    finally:
        os.chdir("CouponBondData")

    for sheet_name in sheets:

        if sheet_name in ["Data", "Temp", "", " "]:
            continue
        else:
            rating: str = sheet_name.split(" ")[0]
            if rating == "NR":
                continue

            if not os.path.exists(rating):
                os.makedirs(rating)

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

                
if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    # Getting the rating of the bonds from an excel sheet path
    ratings = organize_data("Temp_Bloomberg_Data.xlsx")
    
    # Cleaning the folders by removing the files that are not needed
    clean_folders()




    

