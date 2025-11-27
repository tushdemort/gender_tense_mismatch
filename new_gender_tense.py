import argparse
import sys
import json
from typing import Dict, Set, Tuple, Optional, List

import pandas as pd
from tqdm import tqdm

import stanza
from indicnlp import common
from indicnlp.tokenize import indic_tokenize
from indicnlp.morph import unsupervised_morph

# Set path to Indic NLP Resources
common.set_resources_path("indic_nlp_resources-master")


def ensure_hi_pipeline():
    """
    Hindi Stanza pipeline with tokenization disabled (we'll give tokens ourselves),
    and POS+lemma so we can read UPOS + feats (Tense, Gender, etc.)
    """
    try:
        stanza.download('hi')
        nlp = stanza.Pipeline(
            lang="hi",
            processors="tokenize,pos,lemma",
            tokenize_pretokenized=True,  # we give token lists
            tokenize_no_ssplit=True,
            use_gpu=False,
            verbose=False,
        )
        return nlp
    except Exception as e:
        sys.stderr.write(
            "Failed to initialize Stanza Hindi pipeline.\n"
            "Make sure you have installed stanza and downloaded 'hi':\n"
            "  pip install stanza\n"
            "  python -c \"import stanza; stanza.download('hi')\"\n"
            f"Error: {e}\n"
        )
        sys.exit(1)


def tokenize_hindi(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    text = text.strip()
    if not text:
        return []
    return indic_tokenize.trivial_tokenize(text)


def get_roots(tokens: List[str]) -> List[str]:
    """
    Use IndicNLP unsupervised_morph to obtain roots for each token.
    If analysis fails, fall back to the surface token.
    """
    roots = []
    for token in tokens:
        try:
            morphs = unsupervised_morph.unsupervised_morph_analyzer('hi', token)
            root = morphs[0]['root'] if morphs else token
        except Exception:
            root = token
        roots.append(root)
    return roots


def parse_feats(feats_str: Optional[str]) -> Dict[str, str]:
    feats = {}
    if not feats_str:
        return feats
    for kv in feats_str.split("|"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            feats[k] = v
    return feats


def morph_signature(
    sentence: str,
    nlp,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    For a sentence, return:
      - verb_sigs: root -> set of signatures like {"T:Past", "A:Progressive"}
      - genders:   root -> set of gender tags {"Masc", "Fem"}
    """
    verb_sigs: Dict[str, Set[str]] = {}
    genders: Dict[str, Set[str]] = {}

    if not isinstance(sentence, str) or not sentence.strip():
        return verb_sigs, genders

    tokens = tokenize_hindi(sentence)
    if not tokens:
        return verb_sigs, genders

    roots = get_roots(tokens)

    # Stanza with tokenize_pretokenized=True expects a list of token lists
    doc = nlp([tokens])

    for sent in doc.sentences:
        for w in sent.words:
            idx = w.id - 1
            if idx < 0 or idx >= len(roots):
                continue

            root = roots[idx]
            upos = (w.upos or "").upper()
            feats = parse_feats(w.feats)

            # Collect tense/aspect signatures for verbs and auxiliaries
            if upos in {"VERB", "AUX"}:
                sigs: Set[str] = set()

                t = feats.get("Tense")
                if t:
                    t_map = {
                        "Past": "Past",
                        "Pres": "Present",
                        "Fut": "Future",
                        "FutPtc": "Future",
                    }
                    if t in t_map:
                        sigs.add(f"T:{t_map[t]}")

                a = feats.get("Aspect")
                if a:
                    a_map = {
                        "Prog": "Progressive",
                        "Perf": "Perfective",
                        "Imp": "Imperfective",
                        "Hab": "Habitual",
                    }
                    if a in a_map:
                        sigs.add(f"A:{a_map[a]}")

                if sigs:
                    verb_sigs.setdefault(root, set()).update(sigs)

            # Collect gender info for all words
            g = feats.get("Gender")
            if g:
                if "Masc" in g:
                    genders.setdefault(root, set()).add("Masc")
                if "Fem" in g:
                    genders.setdefault(root, set()).add("Fem")

    return verb_sigs, genders


def split_verb_feats(sig_set: Set[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Given a set like {"T:Past", "A:Progressive"}, return (tense, aspect)
    e.g. -> ("Past", "Progressive").
    If a category is missing, returns None for that.
    """
    tense = None
    aspect = None
    for s in sig_set:
        if s.startswith("T:"):
            tense = s[2:]
        elif s.startswith("A:"):
            aspect = s[2:]
    return tense, aspect


def load_synonyms(path: Optional[str]) -> Dict[str, Set[str]]:
    """
    Load a JSON file mapping reference roots to lists of synonym roots:
        {
          "लड़का": ["बालक", "छोरा"],
          "सुंदर": ["खूबसूरत", "सुहावना"]
        }

    Returns a dict[root] = set(synonym_roots).
    If path is None, returns an empty dict.
    """
    if not path:
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        sys.stderr.write(
            f"Failed to load synonyms JSON from {path}: {e}\n"
        )
        return {}

    syn_map: Dict[str, Set[str]] = {}
    for k, v in raw.items():
        if isinstance(v, list):
            syn_map[k] = set(str(x) for x in v)
        else:
            syn_map[k] = {str(v)}

    return syn_map


def get_mt_verb_feats_for_ref_root(
    ref_root: str,
    v_mt: Dict[str, Set[str]],
    synonyms: Dict[str, Set[str]],
) -> Set[str]:
    """
    For a given reference root, collect all verb signatures in MT for:
      - the same root, and
      - any of its synonym roots (if provided).

    Returns the union of all matching MT signature sets.
    """
    sigs: Set[str] = set()

    # Exact same root
    if ref_root in v_mt:
        sigs.update(v_mt[ref_root])

    # Synonym roots (if any)
    for syn_root in synonyms.get(ref_root, set()):
        if syn_root in v_mt:
            sigs.update(v_mt[syn_root])

    return sigs


def get_mt_gender_for_ref_root(
    ref_root: str,
    g_mt: Dict[str, Set[str]],
    synonyms: Dict[str, Set[str]],
) -> Set[str]:
    """
    For a given reference root, collect all genders in MT for:
      - the same root, and
      - any of its synonym roots (if provided).

    Returns the union of all matching MT gender sets.
    """
    genders: Set[str] = set()

    if ref_root in g_mt:
        genders.update(g_mt[ref_root])

    for syn_root in synonyms.get(ref_root, set()):
        if syn_root in g_mt:
            genders.update(g_mt[syn_root])

    return genders


def score_tense(mt: str, ref: str, nlp, synonyms: Dict[str, Set[str]]) -> float:
    """
    Tense+Aspect agreement score in [0,1], reference-anchored.

    For each verb root in the reference:
      - we look at its tense/aspect features in the reference
      - then look at the MT sentence for that same root OR any allowed synonym
      - we check whether MT matches the reference tense and aspect.
    """
    v_mt, _ = morph_signature(mt, nlp)
    v_ref, _ = morph_signature(ref, nlp)

    total_feats = 0  # number of tense/aspect features in reference
    correct = 0      # how many of those MT gets right

    for root, sig_ref in v_ref.items():
        tense_ref, asp_ref = split_verb_feats(sig_ref)
        if tense_ref is None and asp_ref is None:
            # no tense/aspect information on this verb in reference
            continue

        sig_mt = get_mt_verb_feats_for_ref_root(root, v_mt, synonyms)
        tense_mt, asp_mt = split_verb_feats(sig_mt)

        # Tense check
        if tense_ref is not None:
            total_feats += 1
            if tense_mt == tense_ref:
                correct += 1

        # Aspect check
        if asp_ref is not None:
            total_feats += 1
            if asp_mt == asp_ref:
                correct += 1

    if total_feats == 0:
        # No tense/aspect features in reference -> undefined, return 0.0
        return 0.0

    return round(correct / float(total_feats), 4)


def score_gender(mt: str, ref: str, nlp, synonyms: Dict[str, Set[str]]) -> float:
    """
    Gender agreement score in [0,1], reference-anchored.

    For each root in the reference that has a single unambiguous gender:
      - find the same root or any synonym root in MT
      - if MT also has a single gender and it matches the reference, count as correct.
      - otherwise treat as mismatch (missing / ambiguous / different).
    """
    _, g_mt = morph_signature(mt, nlp)
    _, g_ref = morph_signature(ref, nlp)

    total = 0    # number of clearly gendered roots in reference
    correct = 0  # how many MT matches

    for root, ref_genders in g_ref.items():
        # Only consider roots with a single unambiguous gender in reference
        if len(ref_genders) != 1:
            continue

        ref_gender = next(iter(ref_genders))
        total += 1

        mt_genders_all = get_mt_gender_for_ref_root(root, g_mt, synonyms)

        # MT must have exactly one gender, and it must match
        if len(mt_genders_all) == 1 and ref_gender in mt_genders_all:
            correct += 1

    if total == 0:
        # No clear gender-marked roots in reference
        return 0.0

    return round(correct / float(total), 4)


def detect_cols(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Try to infer reference and MT columns from names.
    Fallback: first two columns.
    """
    cols = list(df.columns)
    low = [str(c).lower() for c in cols]
    ref_keys = ["ref", "reference", "gold", "correct", "target", "src", "source"]
    mt_keys = [
        "mt", "translation", "hypothesis", "system", "output",
        "mt_output", "pred", "machine", "hyp"
    ]

    ref_idx = mt_idx = None
    for i, name in enumerate(low):
        if ref_idx is None and any(k in name for k in ref_keys):
            ref_idx = i
        if mt_idx is None and any(k in name for k in mt_keys):
            mt_idx = i

    if ref_idx is None and len(cols) >= 2:
        ref_idx = 0
    if mt_idx is None and len(cols) >= 2:
        mt_idx = 1 if ref_idx != 1 else 0

    return cols[ref_idx], cols[mt_idx]


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Hindi gender/tense agreement scoring using Indic NLP + Stanza.\n"
            "Reference-anchored: checks if MT (or its synonyms) matches "
            "the reference's tense, aspect and gender features on verbs/roots."
        )
    )
    ap.add_argument("--xlsx", required=True, help="Path to Excel file")
    ap.add_argument("--out", required=True, help="Path to output CSV")
    ap.add_argument("--ref-col", help="Reference column name (optional)")
    ap.add_argument("--mt-col", help="Translation/MT column name (optional)")
    ap.add_argument(
        "--synonyms-json",
        help=(
            "Optional path to JSON file mapping reference roots to lists of "
            "synonym roots. These synonyms are treated as acceptable "
            "lexical alternatives and checked for correct tense/gender."
        ),
    )
    args = ap.parse_args()

    nlp = ensure_hi_pipeline()
    df = pd.read_excel(args.xlsx)

    if args.ref_col and args.mt_col:
        ref_col, mt_col = args.ref_col, args.mt_col
    else:
        ref_col, mt_col = detect_cols(df)

    synonyms = load_synonyms(args.synonyms_json)

    refs = df[ref_col].astype(str).fillna("")
    mts = df[mt_col].astype(str).fillna("")

    rows: List[Tuple[str, str, float, float]] = []

    for mt, ref in tqdm(zip(mts, refs), total=len(df), desc="Scoring"):
        t = score_tense(mt, ref, nlp, synonyms)
        g = score_gender(mt, ref, nlp, synonyms)
        rows.append((ref, mt, t, g))

    out_df = pd.DataFrame(
        rows,
        columns=["reference", "translation", "tense_score", "gender_score"]
    )
    out_df.to_csv(args.out, index=False, encoding="utf-8")

    print(f"Saved: {args.out}")
    print(f"Columns used -> reference: {ref_col} | translation: {mt_col}")
    if args.synonyms_json:
        print(f"Synonyms file used: {args.synonyms_json}")


if __name__ == "__main__":
    main()
