# JobScrape

Simple, configurable job scraping for multiple cities from LinkedIn and Indeed using the JobSpy library. It saves a per‑city CSV, merges them into a single Excel workbook with one sheet per city plus an “All” sheet, and cleans up the CSVs.

## Features
- Multi‑city runs from a single command (order controlled by `config.py`).
- Scrapes LinkedIn and Indeed via JobSpy.
- Optional filters by job level (applied to LinkedIn rows) and column slimming.
- Outputs `Jobs/all_jobs.xlsx` with sheets: `All`, then one per city.

## Requirements
- Python 3.10+
- `pip install -r requirements.txt`
- OS: Windows/macOS/Linux (no browser required; JobSpy uses HTTP requests)

## Quick Start
1. Configure the cities and options in `config.py` (see below).
2. Install dependencies:
   - `python -m venv .venv && .venv\Scripts\activate` (Windows) or `python -m venv .venv && source .venv/bin/activate` (macOS/Linux)
   - `pip install -r requirements.txt`
3. Run all cities:
   - `python run_all.py`
4. Open the result at `Jobs/all_jobs.xlsx`.

## Configuration (`config.py`)
- `OUTPUT_DIR`: Folder for outputs (default `Jobs`).
- `CITIES`: Mapping of display name → location string. Display name is used as the Excel sheet name; the location string is passed to JobSpy. The order here controls the sheet order.
- `RESULTS_WANTED`: Max results per site per city.
- `HOURS_OLD`: Only include jobs posted within this many hours.
- `JOB_TYPE`: e.g., `"fulltime"` or `None`.
- `COUNTRY_INDEED`: Country code for Indeed queries (e.g., `"USA"`).
- `LINKEDIN_FETCH_DESCRIPTION`: Whether to fetch job descriptions from LinkedIn.
- `LINKEDIN_JOB_LEVEL_PARAM`: Levels sent to LinkedIn (e.g., entry, associate, mid‑senior).
- `JOB_LEVEL_FILTER`: Post‑merge filter set. If present, it keeps only LinkedIn rows whose `job_level` is in this set (Indeed often lacks this field and is kept).
- `DESIRED_COLUMNS`: If set to a list, keeps only those columns that exist in the scraped data; set to `None` to keep all columns.
- `DEFAULT_LINKEDIN_QUERY` / `DEFAULT_INDEED_QUERY`: Query strings used for each site.

Example city list:
```
CITIES = {
    "Boston": "Boston, MA",
    "Washington DC": "Washington, DC",
    "New York": "New York, NY",
    "Philadelphia": "Philadelphia, PA",
    "Chicago": "Chicago, IL",
}
```

## Usage
- Run all configured cities and merge to Excel:
  - `python run_all.py`
  - Writes per‑city CSVs to `Jobs/`, merges them into `Jobs/all_jobs.xlsx`, then deletes the CSVs.

- Run a single city:
  - `python scrape_city.py --city "Boston"`
  - Optionally override location: `python scrape_city.py --city "Boston" --location "Boston, MA"`

- Output structure:
  - `Jobs/all_jobs.xlsx`: First sheet `All` (all rows combined), then one sheet per city in the same order as `CITIES`.
  - Intermediate CSVs named like `jobs_boston.csv` are automatically removed after merge by `run_all.py`.

## Customizing Queries
Edit `DEFAULT_LINKEDIN_QUERY` and `DEFAULT_INDEED_QUERY` in `config.py`. They support typical Boolean search syntax.

There is an optional helper `query_loader.py` that can parse queries from a Markdown file (headings `## LinkedIn` and/or `## Indeed`). It’s not wired into the scripts by default; if you prefer that workflow, adapt `scrape_city.py:get_queries()` to read from your Markdown instead of `config.py`.

## How It Works
- `scrape_city.py` calls `jobspy.scrape_jobs` for LinkedIn and Indeed with the configured parameters, concatenates results, applies an optional LinkedIn level filter and column slimming, and writes a per‑city CSV into `Jobs/`.
- `run_all.py` iterates cities from `config.CITIES`, runs `scrape_city.py` for each, collects the CSVs, writes `Jobs/all_jobs.xlsx` with `All` + per‑city sheets, then deletes the CSVs.

## Tips
- Fewer filters and a larger `HOURS_OLD` window usually increase results.
- If a site returns few or no results, try relaxing the query string.
- For non‑US roles, adjust `COUNTRY_INDEED` and location strings.

## Troubleshooting
- No results or very few rows:
  - Increase `RESULTS_WANTED` and/or `HOURS_OLD`.
  - Relax Boolean search terms in `config.py`.
- Columns missing:
  - Set `DESIRED_COLUMNS = None` to keep all available fields.
- Excel has only an `Empty` sheet:
  - Indicates no CSVs had data; check console output for per‑site counts.
- LinkedIn/Indeed blocking or errors:
  - Try again later, adjust queries, or update `jobspy` packages to a recent version.

## Project Structure
- `run_all.py`: Orchestrates per‑city scraping and Excel merge, then cleans CSVs.
- `scrape_city.py`: Scrapes a single city from LinkedIn and Indeed using JobSpy.
- `config.py`: Central configuration for cities, queries, and filters.
- `Jobs/`: Output directory (created automatically).
- `personal/`: Personal notes/resume (ignored by Git; not used by scripts).
- `query_loader.py`: Optional helper to parse queries from a Markdown file.

## Notes
- Respect website terms of service and local laws when scraping. Use this tool responsibly.

## GitHub Actions (CI)
You can run the scraper on a schedule or on-demand using GitHub Actions. This repo includes a workflow:

- File: `.github/workflows/scrape.yml`
- Triggers: manual (`workflow_dispatch`) and weekday schedule at 09:00 UTC.
- Output: uploads `Jobs/all_jobs.xlsx` as a build artifact.

Usage:
- Push the repo to GitHub.
- In the GitHub UI, go to Actions → JobScrape → Run workflow to trigger manually, or wait for the schedule.
- Download the `all_jobs` artifact from the workflow run.

Customize:
- Change the cron under `on.schedule` for different time windows.
- Adjust Python version under `actions/setup-python`.
- If you prefer committing the Excel back to the repo, replace the upload step with a commit/push step (add `[skip ci]` to the commit message to avoid triggering a loop), or push to a separate branch.
