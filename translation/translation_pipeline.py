from utils.vietnamese_text_analyzer import VietnameseTextAnalyzer
from utils.reconstruct_sentence import reconstruct_sentence_batch
from utils.word_segmentation import TextSegmenter
from utils.translator import TranslateModel
from utils.search import SearchTranslator
from utils.best_candidate import BestCandidateSelector
from difflib import SequenceMatcher
from config import WORD_PATH

class Translator:
    """Translates a sentence through the full analysis, processing, and reconstruction pipeline."""
    
    def __init__(self, classification_model, translator_model, selector_model, solr_url):
        """
        Initialize Translator.
        """
        self.analyzer = VietnameseTextAnalyzer(word_path=WORD_PATH, model_name=classification_model) 
        self.text_segmenter = TextSegmenter()
        
        self.solr_url = solr_url
        self.search_translator = SearchTranslator(solr_url)
        
        # Initialize models once at startup
        self.translator = TranslateModel(translator_model)
        self.selector = BestCandidateSelector(selector_model)


    def translate(self, sentence):
        test_sentence = sentence
        # Analyze the sentence
        non_foreign_words, remaining_sentence = self.analyzer.analyze_sentence(test_sentence)

        # Print results
        # print(f"Non-foreign words: {non_foreign_words}")
        # print(f"Remaining sentence (marked): {remaining_sentence}")

        segmented_text = self.text_segmenter.segment(remaining_sentence)
        # segmented_text = splitSentenceIntoWords(remaining_sentence)
        # print("Segmented Sentence:", segmented_text)
        words = self.analyzer.normalize_words(segmented_text)
        # print(words)

        # Step 2: Process the sentence in batch
        # print("Processing input sentence...")

        processed_results = self.processSentenceBatch(words, f'{self.solr_url}/select?indent=true&q.op=OR&q=')
        # print(processed_results)

        # Step 3: Reconstruct the sentence
        output_sentence = reconstruct_sentence_batch(processed_results, non_foreign_words)
        # print("Processed Sentence:", output_sentence)
        return output_sentence
    
    
    def processSentenceBatch(self, words, solr_url):
        """
        Process an array of words: look up Solr for dictionary matches and translate
        any words not found using the neural model.
        """
        search_results = self.search_translator.search(words)  # Search Solr for all words
        # print("Dictionary search results:", search_results)

        sentence = ''
        processed_results = []  # Array to store processed results
        i = 0
        non_dict_words = []  # Words not found in the dictionary

        while i < len(words):
            word = words[i]

            # If the word is in <word> form, translate any pending non-dict words first, then keep <word> as-is
            if word.startswith('<') and word.endswith('>'):
                # print(non_dict_words)
                if non_dict_words:
                    temp_combined = ' '.join(non_dict_words)  # Combine all non-dict words
                    temp_translation = self.translator.translate(temp_combined.strip())
                    # print(f"{temp_combined.strip()} -> model translation: {temp_translation}")

                    processed_results.append(temp_translation)
                    sentence += temp_translation + ' '
                    non_dict_words = []  # Reset non-dict word list after translating

                processed_results.append(word)
                sentence += word + ' '
                # print(f"{word} -> kept as-is")
                i += 1
                continue

            best_candidate = None
            best_match_length = 0
            best_combined_word = None

            # Find the longest phrase that can be matched in the dictionary (1 to 4 words)
            for j in range(i, min(i + 4, len(words))):
                combined_word = ' '.join(words[i:j + 1])
                # print(f"Checking phrase: {combined_word}")
                candidates = self.findRelatedCandidates(combined_word, search_results)

                if candidates:
                    best_candidate = self.selector.choose_best_candidate(sentence, candidates)
                    best_match_length = j - i + 1  # Length of the best-matching phrase
                    best_combined_word = combined_word

            # If a best candidate is found, flush pending non-dict words first
            if best_candidate:
                # print(non_dict_words)
                if non_dict_words:
                    temp_combined = ' '.join(non_dict_words)  # Combine all non-dict words
                    temp_translation = self.translator.translate(temp_combined.strip())
                    # print(f"{temp_combined.strip()} -> model translation: {temp_translation}")

                    processed_results.append(temp_translation)
                    sentence += temp_translation + ' '
                    non_dict_words = []  # Reset non-dict word list after translating

                # print(f"{best_combined_word} -> dictionary translation: {best_candidate}")
                processed_results.append(best_candidate)
                sentence += best_candidate + ' '
                i += best_match_length  # Skip over the matched phrase

            else:
                # Word not found in dictionary; add to list for batch model translation
                non_dict_words.append(word)
                i += 1

        # Translate any remaining non-dict words after the loop ends
        if non_dict_words:
            # print(non_dict_words)
            temp_combined = ' '.join(non_dict_words)  # Combine all non-dict words
            temp_translation = self.translator.translate(temp_combined.strip())
            # print(f"{temp_combined.strip()} -> model translation: {temp_translation}")

            processed_results.append(temp_translation)
            sentence += temp_translation + ' '

        return processed_results

    def similarity_ratio(self, a, b):
        """
        Compute the similarity ratio between strings a and b, returning a value between 0 and 1.
        """
        a = a.replace('_', ' ')  # Strip underscores
        b = b.replace('_', ' ')  # Strip underscores
        return SequenceMatcher(None, a, b).ratio()

    def findRelatedCandidates(self, word, search_results):
        """
        Find related Vietnamese candidates for a given word by fuzzy-matching against Solr results.
        Returns candidates whose Bahnar phrase matches the query with at least 85% similarity.
        """
        related_candidates = []
        for result in search_results:
            if 'bahnar' in result and 'vietnamese' in result:
                bahnar_phrase = result['bahnar']
                vietnamese_candidates = result['vietnamese']

                # Check if the query word matches the Bahnar phrase with sufficient similarity
                similarity = self.similarity_ratio(word, bahnar_phrase)
                if similarity >= 0.85:  # Accept matches at 85% similarity or above
                    related_candidates.extend(vietnamese_candidates)

        return related_candidates

