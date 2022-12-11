import pandas as pd
import numpy as np
from sklearn import preprocessing
from autocorrect import Speller

def open_data(file_name, labels):
    df = pd.read_excel(io=file_name)
    df = df[~df['age'].isna()]

    print(len(df))

    if labels and labels != []:
        for item in df:
            if 'nlp' in item and 'label' not in item:
                for label in labels:
                    df = df[df[f'{item}_label'] != label]

    print(len(df))
    
    df['sex'] = [(male, female) for male, female in zip(df['is_male'], df['is_female'])]
    df['nlp_2_4_5'] = [f'{nlp_2}. {nlp_4} .{nlp_5}' for nlp_2, nlp_4, nlp_5 in zip(df['nlp_2'], df['nlp_4'], df['nlp_5'])]
    
    # age_arr = df['age'].values
    # age_norm = preprocessing.normalize([age_arr])
    # df['age_norm'] = age_norm[0]

    # hours_arr = df['hours'].values
    # hours_norm = preprocessing.normalize([hours_arr])
    # df['hours_norm'] = hours_norm[0]

    return df

def choose_data(x, y, labels, file_name):
    df = open_data(file_name, labels)
    df = df.filter([*x, y])
    for item in x:
        df = df[~df[item].isna()]
    print(len(df))
    return df

def get_X_Y(x_data, y_data, transformer_, df_):
    
    all_x_processed = []

    for item in x_data:
        print(item)
        if 'nlp' in item:

            item_processed = transformer_.transform(list(df_[item]))
            all_x_processed.append(item_processed)

        elif item == 'sex':
            all_x_processed.append(list(df_[item]))
        
        else:
            item_processed = [[item] for item in df_[item]]
            all_x_processed.append(item_processed)

    all_x_processed = tuple(all_x_processed)
    x_processed = np.hstack(all_x_processed)
    y_processed = df_[y_data]

    return x_processed, y_processed