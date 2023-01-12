import os

import pandas as pd

from uuid import UUID, uuid4


# nb this is a general function that works only with XTM data format (e.g. Excel sheet with two columns - source_lang, target_lang) 

def merge_data(data_path, save_path, save_name, batch_no):
    """
    output: saves concatenation of all TMs
    :input data_path: string of data directory that contains only XTM formatted Excel sheets
    :input save_path: string of path to save data
    :input batch_no: data version
    """
    
    df = pd.DataFrame(columns=['source', 'target'])
    
    files = os.listdir(data_path)

    for file in files:
        D = pd.read_excel(data_path + file)
        D.columns = ['source', 'target']

        df = pd.concat([df, D])

    df.drop_duplicates(inplace=True)

    df = df[~df['source'].isnull()]
    df = df[~df['target'].isnull()]

    df['id'] = None
    df['batch'] = batch_no

    df['id'] = df['id'].apply(lambda x: uuid4())

    df.to_csv(save_path + save_name, index=False)
    
    return None
