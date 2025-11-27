# !{sys.executable} -m pip install indic-nlp-library
from indicnlp import common
from indicnlp.tokenize import indic_tokenize
from indicnlp.morph import unsupervised_morph
import numpy as np

common.set_resources_path("indic_nlp_resources-master")
import pandas as pd

wordOrd_df = pd.read_csv("gender_tense_out.csv")
print(wordOrd_df.head())
def tokenize_hindi(text: str):
    if not isinstance(text, str):
        return []
    return indic_tokenize.trivial_tokenize(text.strip())

def get_roots(tokens):
    roots = []
    for token in tokens:
        try:
            morphs = unsupervised_morph.unsupervised_morph_analyzer('hi', token)
            root = morphs[0]['root'] if morphs else token
        except Exception:
            root = token
        roots.append(root)
    return roots

wordOrd_df['ref_tokens'] = wordOrd_df['reference'].apply(tokenize_hindi)
wordOrd_df['mt_tokens']  = wordOrd_df['translation'].apply(tokenize_hindi)
wordOrd_df[['reference', 'ref_tokens', 'translation', 'mt_tokens']].head()
wordOrd_df['ref_roots'] = wordOrd_df['ref_tokens'].apply(get_roots)
wordOrd_df['mt_roots']  = wordOrd_df['mt_tokens'].apply(get_roots)
wordOrd_df[['ref_tokens', 'ref_roots', 'mt_tokens', 'mt_roots']].head()
# # Standard Algorithm
# import numpy as np

# def compute_drule_from_roots(mt_roots, ref_roots):
#     drule = None

#     if not isinstance(mt_roots, (list, tuple)) or not isinstance(ref_roots, (list, tuple)):
#         return None

#     aligned_pairs = []
#     for i, ref_root in enumerate(ref_roots):
#         if ref_root in mt_roots:
#             j = mt_roots.index(ref_root)
#             aligned_pairs.append((i, j))

#     if not aligned_pairs:
#         return None

#     penalties = []
#     for ref_pos, mt_pos in aligned_pairs:
#         pos_diff = abs(ref_pos - mt_pos)

#         if pos_diff == 0:
#             p_i = 0
#         elif pos_diff <= 2:
#             p_i = 0.25
#         else:
#             p_i = 0.5

#         penalties.append(p_i)

#     if all(p == 0 for p in penalties):
#         return None

#     p_avg = np.mean(penalties)
#     drule = round(1 - p_avg, 3)

#     return drule

# wordOrd_df['drule'] = wordOrd_df.apply(
#     lambda row: compute_drule_from_roots(row['mt_roots'], row['ref_roots']),
#     axis=1
# )
# wordOrd_df[['ref_roots', 'mt_roots', 'drule']].head(10)

# wordOrd_df['drule'].isna().sum()
# wordOrd_df['drule'].describe()
# Modified Algorithm
import numpy as np

def compute_drule_from_roots2(mt_roots, ref_roots):
    drule = None

    if not isinstance(mt_roots, (list, tuple)) or not isinstance(ref_roots, (list, tuple)):
        return None

    aligned_pairs = []
    for i, ref_root in enumerate(ref_roots):
        if ref_root in mt_roots:
            j = mt_roots.index(ref_root)
            aligned_pairs.append((i, j))

    if not aligned_pairs:
        return None

    aligned_pairs.sort(key=lambda x: x[0])
    phrase_units = []
    current_unit = [aligned_pairs[0]]

    for (prev_ref, prev_mt), (cur_ref, cur_mt) in zip(aligned_pairs, aligned_pairs[1:]):
        if cur_ref == prev_ref + 1 and cur_mt == prev_mt + 1:
            current_unit.append((cur_ref, cur_mt))
        else:
            phrase_units.append(current_unit)
            current_unit = [(cur_ref, cur_mt)]
    phrase_units.append(current_unit)


    word_penalties = []
    for ref_pos, mt_pos in aligned_pairs:
        pos_diff = abs(ref_pos - mt_pos)
        if pos_diff == 0:
            p_i = 0
        elif pos_diff <= 2:
            p_i = 0.25
        else:
            p_i = 0.5
        word_penalties.append(p_i)

    phrase_penalties = []
    for unit in phrase_units:

        if len(unit) > 1:
            ref_positions = [r for r, _ in unit]
            mt_positions = [m for _, m in unit]
            ref_avg = np.mean(ref_positions)
            mt_avg = np.mean(mt_positions)
            displacement = abs(ref_avg - mt_avg)

            preserved = all(mt_positions[i] < mt_positions[i+1] for i in range(len(mt_positions)-1))

            if preserved:
                if displacement <= 2:
                    p_Uk = 0.125  # local intact move
                else:
                    p_Uk = 0.25   # long-range intact move
                phrase_penalties.append(p_Uk)
        else:
            continue


    all_penalties = word_penalties + phrase_penalties
    if all(p == 0 for p in all_penalties):
        return None

    p_avg = np.mean(all_penalties)
    drule = round(1 - p_avg, 3)

    return drule

wordOrd_df['drule'] = wordOrd_df.apply(
    lambda row: compute_drule_from_roots2(row['mt_roots'], row['ref_roots']),
    axis=1
)
wordOrd_df[['ref_roots', 'mt_roots', 'drule']].head(10)

wordOrd_df['drule'].isna().sum()
wordOrd_df['drule'].describe()
finall = wordOrd_df[['reference','translation','tense_score','gender_score','drule']]
finall.to_csv('final.csv')