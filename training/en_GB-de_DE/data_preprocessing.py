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
    
    #Language rules defined from feeback (need to link to confluence page)
    def sie_check(s):
        """
        Output: Boolean check if 'sie'is in the second position
        :param s: string
        """
        words = s.split()
        
        if len(words) > 1:
            if str(s).split()[1] == 'sie':
                return True
            else:
                return False
        else:
            return False
        
        
    dataframe['de_DE'] = dataframe['de_DE'].apply(lambda s: s.replace(
        'Du hast', 'Sie haben').replace(
        'Eure', 'Deine').replace(
        'Euren', 'Deinen').replace(
        'Eurem', 'Deinem').replace(
        'Eurer', 'Deiner').replace(
        'Euer', 'Dein').replace(
        'eure', 'deine').replace(
        'euren', 'deinen').replace(
        'eurem', 'deinem').replace(
        'eurer', 'deiner').replace(
        'euer', 'dein')
                                                 )
    
    dataframe['sie'] = dataframe['de_DE'].apply(lambda s: sie_check(s))
    dataframe = dataframe[dataframe['sie'] == False]                                                  
                                                
    return dataframe[['en_GB', 'de_DE']]

