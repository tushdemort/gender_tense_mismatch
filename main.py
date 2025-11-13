import argparse
import sys
import re
from typing import Dict, Set, Tuple, Optional, List
import pandas as pd
import stanza 
from tqdm import tqdm

stanza.download('hi')


def ensure_hi_pipeline():
    """
    Build a Stanza pipeline for Hindi with tokenization + POS + lemma + MWT.
    Assumes 'stanza.download("hi")' has been run by the user.
    """
    try:
        nlp = stanza.Pipeline(
            lang="hi",
            processors="tokenize,pos,lemma",
            tokenize_no_ssplit=True,  
            use_gpu=False,
            verbose=False,
        )
        return nlp
    except Exception as e:
        sys.stderr.write(
            "Failed to initialize Stanza Hindi pipeline. "
            "Run:\n  pip install stanza pandas openpyxl\n"
            '  python -c "import stanza; stanza.download(\'hi\')"\n'
            f"Error: {e}\n"
        )
        sys.exit(1)

def parse_feats(feats_str: Optional[str]) -> Dict[str, str]:
    feats = {}
    if not feats_str:
        return feats
    for kv in feats_str.split("|"):
        if "=" in kv:
            k, v = kv.split("=", 1)
            feats[k] = v
    return feats

def morph_signature(sentence: str, nlp) -> Tuple[
    Dict[str, Set[str]],  
    Dict[str, Set[str]]   
]:
    verb_sigs: Dict[str, Set[str]] = {}
    genders: Dict[str, Set[str]] = {}
    if not isinstance(sentence, str) or not sentence.strip():
        return verb_sigs, genders

    doc = nlp(sentence)
    for sent in doc.sentences:
        for w in sent.words:
            upos = (w.upos or "").upper()
            lemma = (w.lemma or w.text or "").strip()
            feats = parse_feats(w.feats)

            if upos in {"VERB", "AUX"}:
                sigs: Set[str] = set()
                t = feats.get("Tense")
                if t:
                    t_map = {"Past": "Past", "Pres": "Present", "Fut": "Future", "FutPtc": "Future"}
                    if t in t_map:
                        sigs.add(f"T:{t_map[t]}")
                a = feats.get("Aspect")
                if a:
                    a_map = {
                        "Prog": "Progressive",
                        "Perf": "Perfective",
                        "Imp":  "Imperfective",
                        "Hab":  "Habitual",
                    }
                    if a in a_map:
                        sigs.add(f"A:{a_map[a]}")
                if sigs:
                    verb_sigs.setdefault(lemma, set()).update(sigs)

            g = feats.get("Gender")
            if g:
                if "Masc" in g:
                    genders.setdefault(lemma, set()).add("Masc")
                if "Fem" in g:
                    genders.setdefault(lemma, set()).add("Fem")

    return verb_sigs, genders


def score_tense(mt: str, ref: str, nlp) -> float:
    """
    Prefer per-lemma comparison (same lemma in both sentences).
    If no comparable lemmas, fall back to sentence-level Jaccard over tense/aspect signatures.
    Returns in [-1, 1].
    """
    v_mt, _ = morph_signature(mt, nlp)
    v_rf, _ = morph_signature(ref, nlp)

    common = list(v_mt.keys() & v_rf.keys())
    matches = mismatches = 0

    for lem in common:
        s_mt, s_rf = v_mt.get(lem, set()), v_rf.get(lem, set())
        if not s_mt and not s_rf:
            continue
        if s_mt == s_rf:
            matches += 1
        else:
            mismatches += 1

    if matches + mismatches > 0:
        return round((matches - mismatches) / float(matches + mismatches), 4)

    S_mt = set().union(*v_mt.values()) if v_mt else set()
    S_rf = set().union(*v_rf.values()) if v_rf else set()
    if not S_mt and not S_rf:
        return 0.0
    inter = len(S_mt & S_rf)
    union = len(S_mt | S_rf)
    j = inter / union if union else 0.0
    return round(2 * j - 1, 4)

def score_gender(mt: str, ref: str, nlp) -> float:
    """
    Compare Gender per lemma across both sentences; skip lemmas that are ambiguous (both Masc & Fem) on either side.
    Returns in [-1, 1].
    """
    _, g_mt = morph_signature(mt, nlp)
    _, g_rf = morph_signature(ref, nlp)

    def single_gender(d: Dict[str, Set[str]]) -> Dict[str, str]:
        out = {}
        for lem, gs in d.items():
            if len(gs) == 1:
                out[lem] = next(iter(gs))
        return out

    s_mt = single_gender(g_mt)
    s_rf = single_gender(g_rf)
    common = list(s_mt.keys() & s_rf.keys())
    if not common:
        return 0.0

    matches = sum(1 for lem in common if s_mt[lem] == s_rf[lem])
    mismatches = len(common) - matches
    return round((matches - mismatches) / float(matches + mismatches), 4)


def detect_cols(df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df.columns)
    low = [str(c).lower() for c in cols]
    ref_keys = ["ref", "reference", "gold", "correct", "target"]
    mt_keys  = ["mt", "translation", "hypothesis", "system", "output", "mt_output", "pred", "machine"]
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
    ap = argparse.ArgumentParser(description="Hindi gender/tense mismatch scoring with Stanza (no word lists).")
    ap.add_argument("--xlsx", required=True, help="Path to Excel file")
    ap.add_argument("--out", required=True, help="Path to output CSV")
    ap.add_argument("--ref-col", help="Reference column name (optional)")
    ap.add_argument("--mt-col", help="Translation/MT column name (optional)")
    args = ap.parse_args()

    nlp = ensure_hi_pipeline()
    df = pd.read_excel(args.xlsx)

    ref_col, mt_col = (args.ref_col, args.mt_col) if (args.ref_col and args.mt_col) else detect_cols(df)

    refs = df[ref_col].astype(str).fillna("")
    mts  = df[mt_col].astype(str).fillna("")

    rows: List[Tuple[str, str, float, float]] = []
    for mt, ref in tqdm(zip(mts, refs),total=len(df),desc='Creating Scores'):
        # print('doing tense')
        t = score_tense(mt, ref, nlp)
        # print('doing gender')
        g = score_gender(mt, ref, nlp)
        rows.append((ref, mt, t, g))

    out_df = pd.DataFrame(rows, columns=["reference", "translation", "tense_score", "gender_score"])
    out_df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Saved: {args.out}")
    print(f"Columns used -> reference: {ref_col} | translation: {mt_col}")

if __name__ == "__main__":
    main()
