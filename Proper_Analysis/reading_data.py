import warnings
import pandas as pd
import os

from Proper_Analysis.helpers import connect_answers, normalize_dataset

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

df['NLP_gaming'] = None

df['NLP_well_being'] = None

df = df.drop(0)

connect_answers(df['NLP_all'], df['NLP_1'], df['NLP_2'], df['NLP_3'], df['NLP_4'], df['NLP_5'], df['NLP_6'])
connect_answers(df['NLP_gaming'], df['NLP_2'], df['NLP_3'], df['NLP_4'])
connect_answers(df['NLP_well_being'], df['NLP_1'], df['NLP_2'])

df['GDT_normalized'] = normalize_dataset(df['GDT_score'])

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