import os
from pathlib import Path
from typing import List, Tuple

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

    # Load jobs
    xls = pd.ExcelFile(jobs_excel)
    sheet_name = "All" if "All" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet_name)

    if df.empty:
        # Write an empty file to avoid failing the workflow
        with pd.ExcelWriter(out_excel) as writer:
            df.to_excel(writer, sheet_name="Ranked", index=False)
        print("No jobs to rank. Wrote empty Ranked sheet.")
        return

    # Compose job texts
    def row_to_text(row: pd.Series) -> str:
        title = str(row.get("title", ""))
        company = str(row.get("company", ""))
        location = str(row.get("location", ""))
        desc = str(row.get("description", ""))
        # Truncate very long descriptions to keep embedding request lean
        if len(desc) > 4000:
            desc = desc[:4000]
        return normalize_text(
            f"Title: {title}\nCompany: {company}\nLocation: {location}\nDescription: {desc}"
        )

    texts = [row_to_text(r) for _, r in df.iterrows()]

    # Try OpenAI embeddings; fallback locally if quota/keys missing
    method_used = ""
    try:
        scores, method_used = rank_with_openai(resume_text, texts)
    except Exception as e:
        # Detect insufficient quota or 429 errors to inform the user
        msg = str(e).lower()
        if "insufficient_quota" in msg or "429" in msg or "rate" in msg or "openai_api_key" in msg:
            print(
                "OpenAI embeddings unavailable (quota/key/rate). Falling back to local ranking..."
            )
        else:
            print(f"OpenAI embedding failed: {e}. Falling back to local ranking...")
        scores, method_used = rank_with_local(resume_text, texts)

    df_out = df.copy()
    df_out.insert(0, "similarity_score", scores)
    df_sorted = df_out.sort_values(by="similarity_score", ascending=False)

    # Write output: keep the original sheet(s) and add Ranked
    with pd.ExcelWriter(out_excel) as writer:
        df_sorted.to_excel(writer, sheet_name="Ranked", index=False)
        # Also include source sheet for convenience
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(
        f"Ranked {len(df_sorted)} jobs by similarity using {method_used}. Resume source: {source}.\n"
        f"Wrote {out_excel}"
    )


if __name__ == "__main__":
    main()
