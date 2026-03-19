from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class BestCandidateSelector:
    """
    Selects the best translation candidate based on context
    using the NlpHUST/gpt2-vietnamese language model.
    """

    def __init__(self, model_name="NlpHUST/gpt2-vietnamese", device=None):
        # Determine device (CPU/GPU)
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

    def choose_best_candidate(self, context, candidates):
        """
        Choose the most contextually appropriate candidate.

        Args:
            context (str): The sentence context preceding the word to select.
            candidates (list): List of candidate translations.

        Returns:
            str: The best candidate, or "" if no valid candidates exist.
        """
        if not candidates:  # Return "" for an empty list
            return ""

        candidate_scores = {}

        for candidate in candidates:
            if not candidate:  # Skip empty candidates
                continue

            # Concatenate context with the candidate
            input_text = f"{context} {candidate}"
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

            # Compute probability for the candidate token
            with torch.no_grad():
                outputs = self.model(input_ids)

            # Get logits for the last token
            next_token_logits = outputs.logits[:, -1, :]

            # Score the last token
            score = next_token_logits[0, input_ids[0, -1]].item()

            # Store the candidate score
            candidate_scores[candidate] = score

        # Return "" if no valid candidates were scored
        if not candidate_scores:
            return ""

        # Return the candidate with the highest score
        return max(candidate_scores, key=candidate_scores.get)