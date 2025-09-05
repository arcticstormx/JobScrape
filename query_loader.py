from __future__ import annotations

from typing import Dict


def load_queries(md_path: str) -> Dict[str, str]:
    """
    Extract queries from a Markdown file.
    Expected headings (case-insensitive):
      - ## LinkedIn or ## LinkedIn query
      - ## Indeed or ## Indeed query

    Returns dict with keys 'linkedin' and/or 'indeed' when found.
    """
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
    except FileNotFoundError:
        return {}

    queries: Dict[str, str] = {}
    current: str | None = None

    def norm_head(s: str) -> str:
        s = s.strip().lower()
        if s.startswith("##"):
            s = s.lstrip("# ")
        return s

    for i, line in enumerate(lines):
        h = norm_head(line)
        if h.startswith("linkedin"):
            current = "linkedin"
        elif h.startswith("indeed"):
            current = "indeed"
        else:
            continue

        # take the first non-empty, non-heading line below as the query
        for j in range(i + 1, len(lines)):
            nxt = lines[j].strip()
            if not nxt:
                continue
            if nxt.startswith("##"):
                # next heading encountered without a query
                break
            queries[current] = nxt
            break

    return queries

