import sys
sys.path.insert(0, 'utils')

from config import SOLR_URL, WORD_PATH, CORPUS

print("=== SearchTranslator (with Solr) ===")
from search import SearchTranslator
st = SearchTranslator(SOLR_URL)
print("Solr available:", st.available)
result = st.search(['hrơi', 'ana', 'ngă'])
print("Search result:", result)

print("\n=== Full Translator pipeline ===")
from translation_pipeline import Translator
translator = Translator(
    'undertheseanlp/vietnamese-ner-v1.4.0a2',
    'IAmSkyDra/BartBanaFinal',
    'NlpHUST/gpt2-vietnamese',
    SOLR_URL
)
sentences = [
    'hrơi ana ngă tơdrong pơlei',
    "Sở nông nghiệp păng tơ iung pơ lei, hơ kom, pơ tâng tỉnh",
]
for s in sentences:
    out = translator.translate(s)
    print(f"  IN:  {s}")
    print(f"  OUT: {out}\n")
