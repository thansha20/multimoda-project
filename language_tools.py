# language_tools.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

try:
    print("Initializing Language Model...")
    # Using a common NMT model for translation/Lang ID
    MODEL_NAME = "Helsinki-NLP/opus-mt-mul-en" 
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Language Model loaded successfully on {device}.")
except Exception as e:
    print(f"[CRITICAL ERROR] Language Model failed to load: {e}")
    # We allow the app to run, but this function will fail gracefully
    tokenizer, model = None, None

def identify_and_translate_text(text_input):
    """
    Translates text to English and provides the detected language.
    """
    if not model or not tokenizer:
        return text_input, "Language Model Failed to Load"
        
    if not text_input.strip():
        return "No Input", "N/A"
    
    try:
        # 1. Tokenize for Language Identification
        # The model identifies the source language by its tokenization format
        inputs = tokenizer(text_input, return_tensors="pt", padding=True, truncation=True)
        
        # Extract the language code from the input token IDs (heuristic)
        lang_status = "Unknown"
        if inputs['input_ids'].shape[1] > 1:
             # Heuristic for determining source language code from the special language tokens
            lang_token = inputs['input_ids'][0][1].item() 
            lang_status = tokenizer.convert_ids_to_tokens(lang_token).replace('>>', '').upper()
        
        # 2. Generate Translation
        translation = model.generate(**inputs.to(model.device), max_length=128)
        translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)
        
        return translated_text, lang_status

    except Exception as e:
        print(f"[Language Tool Error]: {e}")
        return text_input, "Translation Failed"