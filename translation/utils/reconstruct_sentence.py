def reconstruct_sentence_batch(processed_results, non_foreign_words):
    """
    Reassemble a sentence from processed tokens in order.
    Replaces <word> placeholders with words from non_foreign_words.
    Capitalizes the start of the sentence and after periods.
    """
    reconstructed_sentence = []
    non_foreign_index = 0
    capitalize_next = True  # Flag to track when to capitalize

    # Reassemble sentence from processed_results and non_foreign_words in order
    for word in processed_results:
        # If the word is <word>, replace it with the next entry from non_foreign_words
        if word == '<word>':
            if non_foreign_index < len(non_foreign_words):
                reconstructed_sentence.append(non_foreign_words[non_foreign_index])
                non_foreign_index += 1
            else:
                reconstructed_sentence.append(word)
        else:
            # Capitalize if at the start of a sentence
            if capitalize_next:
                reconstructed_sentence.append(word.capitalize())  # Capitalize first letter
                capitalize_next = False  # After capitalizing, revert to lowercase
            else:
                reconstructed_sentence.append(word.lower())  # Lowercase remaining words

        # Update capitalize flag after a period
        if word.endswith('.'):
            capitalize_next = True

    # Join tokens into a single string
    reconstructed_sentence = " ".join(reconstructed_sentence).strip()

    # Ensure sentence starts with a capital letter and ends with a period
    if reconstructed_sentence:
        if not reconstructed_sentence[0].isupper():
            reconstructed_sentence = reconstructed_sentence[0].capitalize() + reconstructed_sentence[1:]
        if not reconstructed_sentence.endswith('.'):
            reconstructed_sentence += '.'

    return reconstructed_sentence