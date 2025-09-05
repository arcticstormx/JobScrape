import argparse
import csv
import os
import re
from typing import Tuple

import pandas as pd
from jobspy import scrape_jobs

import config


def city_key(name: str) -> str:
    key = name.strip().lower()
    key = re.sub(r"[^a-z0-9]+", "_", key)
    return key.strip("_")


def get_queries() -> Tuple[str, str]:
    """Return LinkedIn and Indeed queries from config only."""
    return config.DEFAULT_LINKEDIN_QUERY, config.DEFAULT_INDEED_QUERY


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape jobs for a city from LinkedIn + Indeed")
    ap.add_argument("--city", required=True, help="City display name (e.g., 'Boston')")
    ap.add_argument("--location", help="Override location string for JobSpy")
    args = ap.parse_args()

    city = args.city
    location = args.location or config.CITIES.get(city) or city

    linkedin_query, indeed_query = get_queries()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    jobs_linkedin = scrape_jobs(
        site_name=["linkedin"],
        search_term=linkedin_query,
        location=location,
        job_type=config.JOB_TYPE,
        job_level=config.LINKEDIN_JOB_LEVEL_PARAM,
        results_wanted=config.RESULTS_WANTED,
        hours_old=config.HOURS_OLD,
        linkedin_fetch_description=config.LINKEDIN_FETCH_DESCRIPTION,
    )

    jobs_indeed = scrape_jobs(
        site_name=["indeed"],
        search_term=indeed_query,
        location=location,
        results_wanted=config.RESULTS_WANTED,
        hours_old=config.HOURS_OLD,
        country_indeed=config.COUNTRY_INDEED,
        job_type=config.JOB_TYPE,
    )

    print(f"{city}: LinkedIn returned {len(jobs_linkedin)} rows")
    print(f"{city}: Indeed returned {len(jobs_indeed)} rows")

    jobs = pd.concat([jobs_linkedin, jobs_indeed], ignore_index=True)

    # Filter by job_level if present
    # Important: Indeed rows often have missing job_level. Only filter LinkedIn rows by level.
    if "job_level" in jobs.columns and config.JOB_LEVEL_FILTER:
        if "site" in jobs.columns:
            li_mask = jobs["site"].astype(str).str.lower().eq("linkedin")
            keep_li = jobs["job_level"].astype(str).str.lower().isin(config.JOB_LEVEL_FILTER)
            jobs = jobs[(~li_mask) | (li_mask & keep_li)]
        else:
            # Fallback: keep rows with missing job_level (likely non-LinkedIn)
            jl = jobs["job_level"]
            keep = jl.isna() | jl.astype(str).str.lower().isin(config.JOB_LEVEL_FILTER)
            jobs = jobs[keep]

    # Optional column slimming
    if config.DESIRED_COLUMNS:
        keep = [c for c in config.DESIRED_COLUMNS if c in jobs.columns]
        if keep:
            jobs = jobs[keep]

    print(f"{city}: Combined total {len(jobs)} jobs after filters")
    print(jobs.head())

    out_name = f"jobs_{city_key(city)}.csv"
    out_path = os.path.join(config.OUTPUT_DIR, out_name)
    jobs.to_csv(out_path, quoting=csv.QUOTE_NONNUMERIC, escapechar="\\", index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
