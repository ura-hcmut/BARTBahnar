class TranslateModel:
    def __init__(self, model_name="IAmSkyDra/BARTBana_Translation"):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        # Load model and tokenizer once at initialization
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def translate(self, word):
        # Translate the input
        inputs = self.tokenizer(word, return_tensors="pt", truncation=True)
        outputs = self.model.generate(inputs["input_ids"])
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
