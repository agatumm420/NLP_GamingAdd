import warnings
import pandas as pd
import os
path = 'C:\\Users\\Lenovo\\Desktop\\GDT_NLP_Analysis'
os.chdir(path)

warnings.simplefilter("ignore")
file_name = 'GDT_data.xlsx'
df = pd.read_excel(io=file_name)

df['GDT_score'] = df['GDT_Matrix_1'] + df['GDT_Matrix_2'] + df['GDT_Matrix_3'] + df['GDT_Matrix_4']

df['GAD_score'] = df['GAD-7_Matrix_1'] + df['GAD-7_Matrix_2'] + df['GAD-7_Matrix_3'] + df['GAD-7_Matrix_4']\
                  + df['GAD-7_Matrix_5'] + df['GAD-7_Matrix_6'] + df['GAD-7_Matrix_7']

df['PHQ_score'] = df['PHQ_Matrix_1'] + df['PHQ_Matrix_2'] + df['PHQ_Matrix_3'] + df['PHQ_Matrix_4'] + \
                  df['PHQ_Matrix_5'] + df['PHQ_Matrix_6'] + df['PHQ_Matrix_7'] + df['PHQ_Matrix_8'] + \
                  df['PHQ_Matrix_9']

df['NLP_all'] = None

df = df.drop(0)


for i in range(len(df['NLP_all'])):
    i = i + 1
    df['NLP_all'][i] = f'{df["NLP_1"][i]} {df["NLP_2"][i]} {df["NLP_3"][i]} {df["NLP_4"][i]} {df["NLP_5"][i]} {df["NLP_6"][i]}'


model_names = {
    "herbert-klej-cased-v1": {
        "tokenizer": "allegro/herbert-klej-cased-tokenizer-v1",
        "model": "allegro/herbert-klej-cased-v1",
    },
    "herbert-base-cased": {
        "tokenizer": "allegro/herbert-base-cased",
        "model": "allegro/herbert-base-cased",
    },
    "herbert-large-cased": {
        "tokenizer": "allegro/herbert-large-cased",
        "model": "allegro/herbert-large-cased",
    },
}