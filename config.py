"""Central configuration for JobScrape.

Edit values here to affect all scripts.
"""

from __future__ import annotations

from pathlib import Path


# Output directory
OUTPUT_DIR = "Jobs"

# Cities to scrape: {Display Name: Location String}
# Order matters for Excel sheet ordering.
CITIES = {
    "Boston": "Boston, MA",
    "Washington DC": "Washington, DC",
    "New York": "New York, NY",
    "Philadelphia": "Philadelphia, PA",
    "Chicago": "Chicago, IL",
}

# Common scrape parameters
RESULTS_WANTED = 15
HOURS_OLD = 24
JOB_TYPE = "fulltime"  # or None
COUNTRY_INDEED = "USA"

# LinkedIn specific
LINKEDIN_FETCH_DESCRIPTION = True
LINKEDIN_JOB_LEVEL_PARAM = [
    "entry level",
    "associate",
    "mid-senior level",
    "not applicable",
]

# Post-merge filter: keep only these job levels if column exists
JOB_LEVEL_FILTER = {"entry level", "associate", "mid-senior level"}

# Optional column slimming: set to a list to keep only these columns, or None to keep all
DESIRED_COLUMNS = [
    "id",
    "site",
    "job_url",
    "job_url_direct",
    "title",
    "company",
    "location",
    "date_posted",
    "job_type",
    "is_remote",
    "job_level",
    "job_function",
    "description",
    "company_industry",
    "company_url",
]

# Queries for LinkedIn and Indeed
DEFAULT_LINKEDIN_QUERY = (
    '(Trading Specialist OR Investment Associate OR (Trading AND Operations) OR "Trader" OR '
    '"Investment Analyst" OR "Equity Analyst" OR "Fixed Income Analyst" OR "Quantitative Trading" OR '
    '"Institutional Trading" OR "Investment Associate") AND (Entry OR Junior OR Analyst OR Associate) '
    'NOT (sales OR insurance OR real estate)'
)
DEFAULT_INDEED_QUERY = (
    '(Trading Specialist OR Trading Operations OR Investment Associate OR (Trading AND Operations) OR '
    '"Investment Analyst" OR "Equity Analyst" OR "Fixed Income Analyst" OR "Quantitative Trading" OR '
    '"Institutional Trading") AND (Entry OR Junior OR Analyst OR Associate) NOT (sales OR insurance OR '
    'real estate) NOT (Director OR Vice President OR Senior OR Principal OR Executive OR Lead OR VP)'
)
