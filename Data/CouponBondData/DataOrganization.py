import pandas as pd
import os
from typing import Set

def get_rating(path: str) -> Set:
    """Getting the rating of the bonds from an excel sheet path

    Args:
        path (str): Path to the excel file
    """
    excel_file: pd.ExcelFile = pd.ExcelFile(path)

    sheets: list[str] = excel_file.sheet_names

    bond_rating_set: Set = set()

    for sheet_name in sheets:

        if sheet_name in ["Data", "Temp", "", " "]:
            continue
        else:
            rating = sheet_name.split(" ")[0]

            if not os.path.exists(rating):
                os.makedirs(rating)

            # Setting the Working Directory
            os.chdir(rating)

            sheet_df = excel_file.parse(sheet_name)
            sheet_df.to_csv(f"{sheet_name}.csv", index=False)

            # Exiting Out of the Directory
            os.chdir("..")

    return bond_rating_set
        


if __name__ == "__main__":
    os.chdir("Data/CouponBondData")

    # Getting the rating of the bonds from an excel sheet path
    ratings = get_rating("Temp_Bloomberg_Data.xlsx")





    

