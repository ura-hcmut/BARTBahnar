import spacy
import re
import requests
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import os

class VietnameseTextAnalyzer:
    def __init__(self, word_path=None, model_name="undertheseanlp/vietnamese-ner-v1.4.0a2", dictionary_folder="data"):
        """
        Initialize with a Vietnamese dictionary and an NER model.
        """
        self.vietnamese_dict = self.load_vietnamese_dictionary(word_path)

        # Create a blank spaCy model for Vietnamese tokenization
        self.nlp_spacy = spacy.blank("vi")

        # Load the NER model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_model = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple")
        
    def download_vietnamese_dictionary(self, url, file_path):
        """
        Download the Vietnamese dictionary from Google Sheets (exported as XLSX).
        """
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            # print(f"Dictionary downloaded and saved to: {file_path}")
        else:
            print(f"Failed to download dictionary. Status code: {response.status_code}")

    def load_vietnamese_dictionary(self, file_path):
        """
        Load the Vietnamese word list from a plain-text file (one word per line).
        """
        with open(file_path, encoding='utf-8') as f:
            return {line.strip().lower() for line in f if line.strip()}


    def is_special_character(self, word):
        """
        Check if a word consists of special characters.
        """
        return bool(re.match(r"[#$%&()*+,-./:;<=>?@\[\]^`{}~]", word))

    def is_number(self, word):
        """
        Check if a word is an integer or decimal number.
        """
        return word.isdigit() or bool(re.match(r"^0*\d+(\.\d+)?$", word))

    def is_date(self, word):
        """
        Check if a word matches a date pattern.
        """
        date_formats = [
            r"^\d{1,2}/\d{1,2}/\d{4}$",  # dd/mm/yyyy
            r"^\d{1,2}-\d{1,2}-\d{4}$",  # dd-mm-yyyy
            r"^\d{4}/\d{1,2}/\d{1,2}$",  # yyyy/mm/dd
            r"^\d{4}-\d{1,2}-\d{1,2}$",  # yyyy-mm-dd
        ]
        return any(re.match(date_format, word) for date_format in date_formats)

    def is_vietnamese_word(self, word):
        """
        Check if a word exists in the Vietnamese dictionary.
        """
        return word.lower() in self.vietnamese_dict

    def analyze_sentence(self, sentence):
        """
        Analyze a sentence by classifying each token and separating non-Bahnaric words.
        """
        # Tokenize sentence with spaCy
        doc = self.nlp_spacy(sentence)
        tokens = [token.text for token in doc]

        results = []
        for word in tokens:
            if self.is_special_character(word):
                results.append((word, "special_character"))
            elif self.is_number(word):
                results.append((word, "number"))
            elif self.is_date(word):
                results.append((word, "date"))
            elif self.is_vietnamese_word(word):
                results.append((word, "vietnamese"))
            else:
                results.append((word, "other_language"))

        # Keep words that are not Bahnaric (other_language)
        non_foreign_words = [word for word, category in results if category != "other_language"]

        # Build the remaining sentence with Bahnaric words intact and others as <word>
        remaining_sentence = " ".join([f"<word>" if category != "other_language" else word
                                       for word, category in results])

        return non_foreign_words, remaining_sentence

    def normalize_words(self, word_list):
        """
        Normalize a list of words (strip underscores and convert to lowercase).
        """
        return [word.replace('_', ' ').lower() for word in word_list]
