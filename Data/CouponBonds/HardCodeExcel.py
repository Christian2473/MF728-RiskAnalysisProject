import openpyxl

wb = openpyxl.load_workbook("Data\CouponBonds\Temp_Bloomberg_Data_Formulas.xlsx", data_only=True)

wb.save("Data\CouponBonds\Temp_Bloomberg_Data.xlsx")


    