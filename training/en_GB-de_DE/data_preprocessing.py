import pandas as pd

from datasets import Dataset, DatasetDict, load_dataset


def process_data():
    """
    Output: Translated text of the source text
    :param src_text: source string
    """ 
    
    #Load data (needs to be changed to loading from BQ or bucket)
    df = pd.read_csv('../../data/en_GB-de_DE/cleaned_data.csv', engine='python')

    df = df[['en_GB', 'de_DE']] # will be redundent when setup 
    df = df.sample(1000)
    # Remove nulls/duplicates
    df = df[(df['en_GB'].notnull()) & (df['de_DE'].notnull())]
    df.drop_duplicates()
    
    df = language_rules(df)

    # Transform to HuggingFace Dataset
    data = Dataset.from_pandas(pd.DataFrame({'translation': df.to_dict('records')})) 

    # Split data into training sets
    train_test_valid = data.train_test_split(shuffle=True, seed=7, test_size=0.0015)
    test_valid = train_test_valid['test'].train_test_split(shuffle=True, seed=7, test_size=0.5)

    # Convert to train/validate/test
    dataset = DatasetDict({
        'train': train_test_valid['train'],
        'validation': test_valid['test'],
        'test': test_valid['train']})
     
    return dataset


def language_rules(dataframe):
    """
    Output: Dataset after applying language rules to improve training
    :param dataframe: dataframe of translation memories
    """  
    return dataframe
