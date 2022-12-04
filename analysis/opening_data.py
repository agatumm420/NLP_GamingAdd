import pandas as pd
import numpy as np

def open_data():
    file_name = 'GDT_NLP_v1_supervision.xlsx'
    df = pd.read_excel(io=file_name)
    df['sex'] = [(male, female) for male, female in zip(df['is_male'], df['is_female'])]
    return df


def choose_data(x, y, labels=[0, 1]):
    df = open_data()

    for item in x:
        df = df[~df[item].isna()]

        if labels:
            if 'nlp' in item:
                for label in labels:
                    df = df[df[f'{item}_label'] != label]

    df = df.filter([*x, y])
    print([*x, y])
    return df

def get_X_Y(x_data, y_data, transformer_, df_):
    
    all_x_processed = []

    for item in x_data:
        print(item)
        if 'nlp' in item:

            item_processed = transformer_.transform(df_[item])
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


X = [
    'sex',
    'hours'
    ]

Y = 'age'

df = choose_data(X, Y, labels=[0, 1])


train_x, train_y = get_X_Y(X, Y, None, df)
