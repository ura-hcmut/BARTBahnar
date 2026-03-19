from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DICTIONARY_PATH = str(REPO_ROOT / "data" / "dictionary" / "bavi.csv")
WORD_PATH = str(REPO_ROOT / "data" / "corpus" / "vietnamese_words.txt")
CORPUS = str(REPO_ROOT / "data" / "corpus" / "bahnaric.txt")
SOLR_URL = "http://localhost:8983/solr/mycore"
TRANSLATOR_MODEL = str(Path(__file__).resolve().parent / "checkpoints" / "BartBanaFinal")
