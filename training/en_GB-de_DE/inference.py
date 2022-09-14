from transformers import MarianMTModel, MarianTokenizer


model_path = "../EU_IKEA_clean_data_2022-06-28 00:03:00.355629/"


def glossary_full_match():
    """
    Output: Translated text of the source text
    :param src_text: source string
    """  
    
    pass


def glossary_dnt(): # need to be a new tokenizer (or add special tokens?) 
    """
    Output: Translated text of the source text
    :param src_text: source string
    """  
    pass

    
def load_model_from_bucket():
    """
    Output: Loads model from GCP BQ or bucket
    """ 
    pass


def inference(src_text, model_path=model_path):
    """
    Output: Translated text of the source text
    :param src_text: source string
    """   
    # model =  load_model_from_bucket()
    
    if glossary_full_match(src_text):
        return src_text
        
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name, output_loading_info=False)
    
    # src_text = glossary_dnt()
    
    translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=False))
    translated_txt = [tokenizer.decode(s, skip_special_tokens=True) for s in translated]

    return translated_txt