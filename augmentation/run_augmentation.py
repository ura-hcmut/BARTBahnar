from pathlib import Path
import os

import pandas as pd
from augmentation_methods import (
    Combine,
    RandomDeletion,
    RandomInsertion,
    ReplaceWithSameSynonyms,
    ReplaceWithSameThemes,
    SlidingWindows,
    SwapSentences,
)

# ===================== CONFIGURATION =====================
# Paths (relative to this file, or use absolute paths)
INPUT_PATH = '../data/raw/train.csv'       # CSV with columns: LANG_SOURCE, LANG_TARGET
DICTIONARY_PATH = 'path/to/your/dict.csv' # Thematic/POS-tagged dictionary — required ONLY for
                                           # methods 3 (Replace theme), 4 (Replace synonym), 5 (Insert).
                                           # Must have columns: LANG_SOURCE, LANG_TARGET, pos, theme.
                                           # NOTE: this is NOT data/dictionary/bavi.csv

# Language column names — must match headers in INPUT_PATH and DICTIONARY_PATH
LANG_SOURCE = 'Bahnaric'
LANG_TARGET = 'Vietnamese'

# Select augmentation method(s) (1–7), or a list to combine several:
#   1 = Combine        2 = Swap           3 = Replace (theme)
#   4 = Replace (syn)  5 = Insert         6 = Delete
#   7 = Sliding Window
# Examples:  METHOD_NUMS = 1   or   METHOD_NUMS = [1, 6, 7]
METHOD_NUMS = 1

# Per-method settings (only the relevant ones are used)
BATCH_SIZE = 10          # Combine: sentences per batch
LIMIT_NEW_SENTENCES = 10 # Replace: max new sentences per original
NUM_INSERTIONS = 1       # Insert: words inserted per sentence
MAX_LINES_GENERATED = 10 # Insert: cap on total output lines
NUM_DELETIONS = 1        # Delete: words removed per sentence
WINDOW_SIZE = 2          # Sliding Window: window size
# =========================================================

_METHOD_NAMES = {
    1: 'combined', 2: 'swapped_sentences', 3: 'replaced_with_same_themes',
    4: 'replaced_with_same_synonyms', 5: 'random_insertion',
    6: 'random_deletion', 7: 'sliding_windows',
}


def _run_single(method_num, input_path, theme_path, base_dir):
    """Run one augmentation method and return the resulting DataFrame."""
    if method_num == 1:
        print("Running Combine...")
        obj = Combine(LANG_SOURCE, LANG_TARGET, input_path, batch_size=BATCH_SIZE)
        return obj.augment(None)
    elif method_num == 2:
        print("Running SwapSentences...")
        obj = SwapSentences(LANG_SOURCE, LANG_TARGET, input_path)
        return obj.augment(None)
    elif method_num == 3:
        print("Running ReplaceWithSameThemes...")
        obj = ReplaceWithSameThemes(LANG_SOURCE, LANG_TARGET, input_path, theme_path, '')
        return obj.augment()
    elif method_num == 4:
        print("Running ReplaceWithSameSynonyms...")
        obj = ReplaceWithSameSynonyms(LANG_SOURCE, LANG_TARGET, input_path, theme_path, '')
        return obj.augment()
    elif method_num == 5:
        print("Running RandomInsertion...")
        obj = RandomInsertion(LANG_SOURCE, LANG_TARGET, input_path, theme_path)
        return obj.augment()
    elif method_num == 6:
        print("Running RandomDeletion...")
        obj = RandomDeletion(LANG_SOURCE, LANG_TARGET, input_path, num_deletions=NUM_DELETIONS)
        return obj.augment(None)
    elif method_num == 7:
        print("Running SlidingWindows...")
        obj = SlidingWindows(LANG_SOURCE, LANG_TARGET, input_path, window_size=WINDOW_SIZE)
        return obj.augment(None)
    else:
        raise ValueError(f"Unknown method number: {method_num}. Choose 1–7.")


def main():
    base_dir = Path(__file__).resolve().parent
    input_path = str(base_dir / INPUT_PATH)
    theme_path = str(base_dir / DICTIONARY_PATH)
    output_dir = base_dir / 'output'
    os.makedirs(output_dir, exist_ok=True)

    methods = METHOD_NUMS if isinstance(METHOD_NUMS, list) else [METHOD_NUMS]

    if len(methods) == 1:
        result = _run_single(methods[0], input_path, theme_path, base_dir)
        out_file = str(output_dir / f"{_METHOD_NAMES[methods[0]]}.csv")
        result.to_csv(out_file, index=False, encoding='utf-8')
        print(f"Saved to {out_file}")
    else:
        parts = []
        for m in methods:
            df = _run_single(m, input_path, theme_path, base_dir)
            parts.append(df)
        result = pd.concat(parts, ignore_index=True).drop_duplicates()
        label = '+'.join(str(m) for m in methods)
        out_file = str(output_dir / f"augmented_{label}.csv")
        result.to_csv(out_file, index=False, encoding='utf-8')
        print(f"\nCombined {len(methods)} methods → {len(result)} rows")
        print(f"Saved to {out_file}")


if __name__ == '__main__':
    main()

