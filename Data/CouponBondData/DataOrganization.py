import pandas as pd
import os
from contextlib import contextmanager

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

    sheets: list[str] = excel_file.sheet_names

    for sheet_name in sheets:

        if sheet_name in ["Data", "Temp", "", " "]:
            continue
        else:
            rating: str = sheet_name.split(" ")[0]

            if not os.path.exists(rating):
                os.makedirs(rating)

            # Changing the Directory
            with open_folder(rating):
                file_path = os.path.join(os.getcwd(), sheet_name + ".csv")

                # Checking if the file already exists
                if not os.path.exists(file_path):
                    sheet_df = excel_file.parse(sheet_name)
                    sheet_df.to_csv(f"{sheet_name}.csv", index=False)

if __name__ == "__main__":
    os.chdir("Data/CouponBondData")

    # Getting the rating of the bonds from an excel sheet path
    ratings = organize_data("Temp_Bloomberg_Data.xlsx")





    

