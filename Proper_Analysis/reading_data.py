import warnings
import pandas as pd

warnings.simplefilter("ignore")
file_name = 'GDT_data.xlsx'
df = pd.read_excel(io=file_name)

df['GDT_score'] = df['GDT_Matrix_1'] + df['GDT_Matrix_2'] + df['GDT_Matrix_3'] + df['GDT_Matrix_4']

df['GAD_score'] = df['GAD-7_Matrix_1'] + df['GAD-7_Matrix_2'] + df['GAD-7_Matrix_3'] + df['GAD-7_Matrix_4']\
                  + df['GAD-7_Matrix_5'] + df['GAD-7_Matrix_6'] + df['GAD-7_Matrix_7']

df = df.drop(0)


