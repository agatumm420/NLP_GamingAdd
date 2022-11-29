import warnings
import pandas as pd
import os

path = 'C:\\Users\\Lenovo\\Desktop\\GDT\\GDT_NLP_Analysis'
os.chdir(path)
warnings.simplefilter("ignore")

file_name = 'GDT_data_v01.xlsx'

df = pd.read_excel(io=file_name)

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