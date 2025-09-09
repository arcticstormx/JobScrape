import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pypdf is required to read PDF resume. Please add it to requirements.txt"
        ) from e
    reader = PdfReader(str(path))
    parts: List[str] = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts)


def load_resume_text(resume_path: Path) -> Tuple[str, str]:
    """Return (source, text). Supports .pdf and text-like files (md, txt)."""
    if resume_path.suffix.lower() == ".pdf":
        return "pdf", read_pdf_text(resume_path)
    else:
        return resume_path.suffix.lower().lstrip("."), read_text_file(resume_path)


def normalize_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = "\n".join(line.strip() for line in s.splitlines())
    # collapse multiple newlines/spaces
    s = " ".join(s.split())
    return s.strip()


def load_rank_instructions() -> Optional[str]:
    """Load private ranking instructions.

    Looks for env var RANK_INSTRUCTIONS, then file at RANK_INSTRUCTIONS_PATH,
    defaulting to personal/rank_instructions.md if present. Returns None if none found.
    """
    instr = os.getenv("RANK_INSTRUCTIONS")
    if instr:
        return normalize_text(instr)
    path = os.getenv("RANK_INSTRUCTIONS_PATH")
    cand = Path(path) if path else Path("personal/rank_instructions.md")
    if cand.exists():
        try:
            return normalize_text(read_text_file(cand))
        except Exception:
            return None
    return None


def build_resume_query(resume_text: str, instructions: Optional[str]) -> str:
    """Combine instructions and resume into a single query string for embedding.

    The instructions are private (e.g., in personal/rank_instructions.md) and are
    simply prepended to influence similarity without being written to outputs.
    """
    if not instructions:
        return resume_text
    return normalize_text(
        f"Ranking instructions: {instructions}\n\nCandidate resume: {resume_text}"
    )


# -------------------------
# Rule-based Rubric Scoring (Ranking AI)
# -------------------------

TOP_PROP_FIRMS = [
    "Citadel",
    "Jane Street",
    "Optiver",
    "IMC",
    "DRW",
    "Hudson River Trading",
    "Jump",
    "Five Rings",
    "SIG",
    "Flow Traders",
    "Two Sigma Securities",
    "Tower",
    "Akuna",
    "Wolverine",
    "Belvedere",
    "CTC",
    "DV Trading",
    "Virtu",
    "XTX",
]

TITLE_KEYWORDS = [
    "assistant trader",
    "execution trader",
    "trading assistant",
    "trading operations",
    "trade support",
    "trading specialist",
    "investment associate",
    "broker dealer",
]

EQUITY_KEYWORDS = ["equities", "options", "etf", "stock markets", "brokerage"]

# Brokerage/Client-Facing Fit (+2)
BROKERAGE_CLIENT_KEYWORDS = [
    "brokerage",
    "broker-dealer",
    "broker dealer",
    "client service",
    "corporate actions",
    "trade support",
    "trading operations",
    "active trader",
]

TRADING_DESK_KEYWORDS = ["trading desk"]

LICENSE_KEYWORDS = ["series 7", "series 63"]

PREFERRED_LOCATIONS = [
    "new york",
    "nyc",
    "chicago",
    "boston",
    "washington dc",
    "d.c.",
    "dc",
]
FLORIDA_LOCATIONS = ["florida", "miami", "tampa", "orlando"]

SENIORITY_KEYWORDS = ["senior", "vp", "director", "principal", "lead"]


def score_job_posting(title: str, company: str, location: str, description: str):
    score = 0
    breakdown: List[str] = []

    title_lc = str(title or "").lower()
    company_lc = str(company or "").lower()
    location_lc = str(location or "").lower()
    desc_lc = str(description or "").lower()

    if any(firm.lower() in company_lc for firm in TOP_PROP_FIRMS):
        score += 4
        breakdown.append("+4 Top Prop Firm")

    if any(keyword in title_lc for keyword in TITLE_KEYWORDS):
        score += 3
        breakdown.append("+3 Title Match")

    if any(keyword in desc_lc for keyword in EQUITY_KEYWORDS):
        score += 2
        breakdown.append("+2 Equities/Markets Focus")

    if any(keyword in desc_lc for keyword in TRADING_DESK_KEYWORDS):
        score += 2
        breakdown.append("+2 Trading Desk")

    # Brokerage/Client-Facing Fit (+2) — check title or description
    if any(kw in title_lc or kw in desc_lc for kw in BROKERAGE_CLIENT_KEYWORDS):
        score += 2
        breakdown.append("+2 Brokerage/Client-Facing Fit")

    if any(keyword in desc_lc for keyword in LICENSE_KEYWORDS):
        score += 1
        breakdown.append("+1 Series License")

    if any(loc in location_lc for loc in PREFERRED_LOCATIONS):
        score += 1
        breakdown.append("+1 Preferred Location")

    if any(keyword in title_lc for keyword in SENIORITY_KEYWORDS):
        score -= 5
        breakdown.append("-5 Seniority Penalty")

    if any(loc in location_lc for loc in FLORIDA_LOCATIONS):
        score -= 1
        breakdown.append("-1 Florida Penalty")

    # New threshold: Top Pick at >= 10 (max now 15)
    top_pick = score >= 10
    return score, breakdown, top_pick


def build_ranking_ai_sheet(all_df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame scored by the rubric, sorted by ai_score desc.

    Adds columns: ai_score (int), ai_top_pick (bool), ai_breakdown (str)
    """
    if all_df.empty:
        return all_df.copy()

    # Remove postings that would incur a seniority penalty entirely
    if "title" in all_df.columns:
        title_lc = all_df["title"].astype(str).str.lower()
        seniority_pat = "|".join(map(re.escape, SENIORITY_KEYWORDS))
        all_df = all_df[~title_lc.str.contains(seniority_pat, na=False)]

    rows = []
    for _, r in all_df.iterrows():
        title = r.get("title", "")
        company = r.get("company", "")
        location = r.get("location", "")
        desc = r.get("description", "")
        score, breakdown, top_pick = score_job_posting(title, company, location, desc)
        rows.append((score, top_pick, "; ".join(breakdown)))

    out = all_df.copy()
    scores, top_picks, breakdowns = zip(*rows) if rows else ([], [], [])
    out.insert(0, "ai_score", list(scores))
    out.insert(1, "ai_top_pick", list(top_picks))
    out.insert(2, "ai_breakdown", list(breakdowns))
    out = out.sort_values(by=["ai_score"], ascending=False)
    return out

def get_openai_client():
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "openai package is required. Please add it to requirements.txt"
        ) from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # No key present; return None to allow local fallback.
        return None
    base_url = os.getenv("OPENAI_BASE_URL")  # optional, supports Azure/other proxies
    client = OpenAI(api_key=api_key, base_url=base_url)
    return client


def embed_texts(client, texts: List[str], model: str) -> List[List[float]]:
    # OpenAI API supports batching multiple inputs in one call
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]


def rank_with_openai(resume_text: str, texts: List[str]) -> Tuple[List[float], str]:
    """Try to rank using OpenAI embeddings. Returns (scores, method_label).

    Raises an exception if embeddings cannot be obtained (e.g., quota).
    """
    client = get_openai_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY not set")
    embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    [resume_vec] = embed_texts(client, [resume_text], embed_model)
    job_vecs = embed_texts(client, texts, embed_model)

    resume_arr = np.array(resume_vec, dtype=float)
    scores = [cosine_sim(resume_arr, np.array(v, dtype=float)) for v in job_vecs]
    return scores, f"openai:{embed_model}"


def rank_with_local(resume_text: str, texts: List[str]) -> Tuple[List[float], str]:
    """Rank using local text similarity without external APIs.

    Tries scikit-learn TF-IDF if available; falls back to Jaccard set similarity.
    Returns (scores, method_label).
    """
    # Attempt TF-IDF via scikit-learn if present
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=50000,
        )
        corpus = [resume_text] + texts
        X = vectorizer.fit_transform(corpus)
        resume_vec = X[0]
        job_vecs = X[1:]
        sims = cosine_similarity(job_vecs, resume_vec)[:, 0]
        return sims.tolist(), "local_tfidf"
    except Exception:
        pass

    # Lightweight fallback: Jaccard similarity on token sets
    def tokenize(s: str) -> set:
        return set(
            t for t in " ".join(s.lower().split()).split(" ") if t and t.isalpha()
        )

    resume_tokens = tokenize(resume_text)
    scores: List[float] = []
    for txt in texts:
        jt = tokenize(txt)
        if not resume_tokens and not jt:
            scores.append(0.0)
            continue
        inter = len(resume_tokens & jt)
        union = len(resume_tokens | jt)
        scores.append(float(inter / union) if union else 0.0)
    return scores, "local_jaccard"


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main():
    # Inputs
    jobs_excel = Path(os.getenv("JOBS_EXCEL", "Jobs/all_jobs.xlsx"))
    out_excel = Path(os.getenv("OUT_EXCEL", "Jobs/all_jobs_ranked.xlsx"))

    # Prefer PDF resume; fallback to markdown
    default_resume_candidates = [
        Path("personal/resume.pdf"),
        Path("personal/resume.md"),
        Path("resume.pdf"),
        Path("resume.md"),
    ]
    resume_path_env = os.getenv("RESUME_PATH")
    resume_path = Path(resume_path_env) if resume_path_env else None
    if resume_path is None:
        for p in default_resume_candidates:
            if p.exists():
                resume_path = p
                break
    if resume_path is None:
        raise FileNotFoundError(
            "Resume not found. Provide RESUME_PATH or add personal/resume.pdf or personal/resume.md"
        )

    if not jobs_excel.exists():
        raise FileNotFoundError(f"Jobs Excel not found at {jobs_excel}")

    source, resume_text = load_resume_text(resume_path)
    resume_text = normalize_text(resume_text)
    instructions = load_rank_instructions()
    resume_query_text = build_resume_query(resume_text, instructions)

    # Load jobs workbook
    xls = pd.ExcelFile(jobs_excel)
    sheet_order = xls.sheet_names
    if not sheet_order:
        with pd.ExcelWriter(out_excel) as writer:
            pd.DataFrame().to_excel(writer, sheet_name="Empty", index=False)
        print("No sheets found. Wrote Empty sheet.")
        return

    # Helper to convert a row to text for embedding
    def row_to_text(row: pd.Series) -> str:
        title = str(row.get("title", ""))
        company = str(row.get("company", ""))
        location = str(row.get("location", ""))
        desc = str(row.get("description", ""))
        if len(desc) > 4000:
            desc = desc[:4000]
        return normalize_text(
            f"Title: {title}\nCompany: {company}\nLocation: {location}\nDescription: {desc}"
        )

    # Function to rank a single DataFrame
    def rank_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        if df.empty:
            return df.copy(), "no_data"
        # Remove postings that would incur a seniority penalty entirely
        if "title" in df.columns:
            title_lc = df["title"].astype(str).str.lower()
            seniority_pat = "|".join(map(re.escape, SENIORITY_KEYWORDS))
            df = df[~title_lc.str.contains(seniority_pat, na=False)]
        texts = [row_to_text(r) for _, r in df.iterrows()]
        method_used = ""
        try:
            scores, method_used = rank_with_openai(resume_query_text, texts)
        except Exception as e:
            msg = str(e).lower()
            if (
                "insufficient_quota" in msg
                or "429" in msg
                or "rate" in msg
                or "openai_api_key" in msg
            ):
                print(
                    "OpenAI embeddings unavailable (quota/key/rate). Falling back to local ranking..."
                )
            else:
                print(f"OpenAI embedding failed: {e}. Falling back to local ranking...")
            scores, method_used = rank_with_local(resume_query_text, texts)

        out = df.copy()
        out.insert(0, "similarity_score", scores)
        out = out.sort_values(by="similarity_score", ascending=False)
        return out, method_used

    # Build an All dataframe from input (prefer the input 'All' if present)
    if "All" in sheet_order:
        input_all_df = pd.read_excel(xls, sheet_name="All")
    else:
        input_all_df = pd.concat(
            [pd.read_excel(xls, sheet_name=s) for s in sheet_order], ignore_index=True
        )

    # Rank each sheet independently and write to output with same sheet names
    methods_used: Dict[str, str] = {}
    with pd.ExcelWriter(out_excel) as writer:
        # First, add the rule-based 'Ranking AI' sheet based on all jobs
        ranking_ai_df = build_ranking_ai_sheet(input_all_df)
        ranking_ai_df.to_excel(writer, sheet_name="Ranking AI", index=False)
        methods_used["Ranking AI"] = "rubric"

        # Ensure 'All' (if exists) is written first to mimic original ordering
        ordered = (
            ["All"] + [s for s in sheet_order if s != "All"]
            if "All" in sheet_order
            else sheet_order
        )
        for name in ordered:
            df_sheet = pd.read_excel(xls, sheet_name=name)
            ranked_df, method = rank_df(df_sheet)
            methods_used[name] = method
            ranked_df.to_excel(writer, sheet_name=name, index=False)

    # Report summary
    method_summary = ", ".join(f"{k}:{v}" for k, v in methods_used.items())
    print(
        f"Ranked workbook '{jobs_excel.name}' with methods per sheet [{method_summary}]. Resume source: {source}.\n"
        f"Wrote {out_excel}"
    )


if __name__ == "__main__":
    main()
