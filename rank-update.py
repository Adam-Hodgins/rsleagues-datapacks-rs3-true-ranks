import json
import math
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter


BASE_URL = (
    "https://secure.runescape.com/m=hiscore_oldschool_seasonal/overall"
    "?category_type=1&table=1&page={page_number}"
)
MAX_PAGE_GUESS = 20000
REQUEST_TIMEOUT_SECONDS = 20
MAX_REQUEST_RETRIES = 4
RETRY_BACKOFF_SECONDS = 2
VERBOSE = False
SHOW_PROGRESS = True
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) "
        "Gecko/20100101 Firefox/128.0"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://secure.runescape.com/m=hiscore_oldschool_seasonal/overall",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

RANK_PERCENTILES = [
    ("Dragon", 0.01),
    ("Rune", 0.10),
    ("Adamant", 0.20),
    ("Mithril", 0.40),
    ("Steel", 0.60),
    ("Iron", 0.80),
    ("Bronze", 1.00),
]

DATA_DIR = Path(__file__).parent / "Data"
OUTPUT_FILENAME = "rank-thresholds.json"

session = requests.Session()
session.headers.update(REQUEST_HEADERS)


def prepare_output_path():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / OUTPUT_FILENAME
    if output_path.exists():
        archive_number = 1
        while (DATA_DIR / f"{output_path.stem}_{archive_number}{output_path.suffix}").exists():
            archive_number += 1
        archive_path = DATA_DIR / f"{output_path.stem}_{archive_number}{output_path.suffix}"
        output_path.rename(archive_path)
    return output_path


def load_previous_output():
    output_path = DATA_DIR / OUTPUT_FILENAME
    if not output_path.exists():
        return None
    try:
        with output_path.open(encoding="utf-8") as output_file:
            return json.load(output_file)
    except (json.JSONDecodeError, OSError):
        return None


def _previous_rank_players(previous_data, rank_name):
    if not previous_data:
        return None
    for rank in previous_data.get("ranks", []):
        if rank.get("name") == rank_name:
            value = rank.get("players_qualified")
            return value if isinstance(value, int) else None
    return None


def _change(current_value, previous_value):
    if previous_value is None:
        return None
    return current_value - previous_value


def _parse_int(raw_value):
    return int(raw_value.replace(",", ""))


def print_progress(step, total_steps, message):
    percent = (step / total_steps) * 100
    print(f"[{step}/{total_steps} | {percent:5.1f}%] {message}")


def print_inline_status(message):
    print(f"\r{message}", end="", flush=True)


def _fetch_hiscores(url):
    last_error = None
    for attempt in range(1, MAX_REQUEST_RETRIES + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return response
        except requests.HTTPError as exc:
            last_error = exc
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code in (403, 429, 500, 502, 503, 504) and attempt < MAX_REQUEST_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise
        except requests.RequestException as exc:
            last_error = exc
            if attempt < MAX_REQUEST_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            raise
    raise last_error


def _parse_hiscores_html(html):
    soup = BeautifulSoup(html, "html.parser")
    parsed_rows = []

    for row in soup.select("tr.personal-hiscores__row"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        rank = _parse_int(cells[0].text.strip())
        score = _parse_int(cells[-1].text.strip())
        parsed_rows.append((rank, score))
    if parsed_rows:
        return tuple(parsed_rows)

    content_div = soup.find("div", {"id": "contentHiscores"})
    if not content_div:
        return tuple()

    table = content_div.find("table")
    if not table:
        return tuple()

    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < 3:
            continue
        rank = _parse_int(cells[0].text.strip())
        score = _parse_int(cells[2].text.strip())
        parsed_rows.append((rank, score))

    return tuple(parsed_rows)


@lru_cache(maxsize=None)
def get_hiscores_page(page_number):
    """
    Returns a tuple of (rank, score) rows for a hiscores page.
    Cached so repeated lookups do not re-fetch pages.
    """
    url = BASE_URL.format(page_number=page_number)
    response = _fetch_hiscores(url)
    return _parse_hiscores_html(response.text)


def get_page_signature(page_number):
    """
    Lightweight fingerprint used to detect "out of range" pages.
    OSRS returns page 1 when page_number is too high.
    """
    rows = get_hiscores_page(page_number)
    if not rows:
        return tuple()
    return rows[0], rows[-1]


def get_total_pages():
    first_page_signature = get_page_signature(1)

    low = 2
    high = MAX_PAGE_GUESS
    last_valid_page = 1
    iteration = 0

    while low <= high:
        iteration += 1
        mid = (low + high) // 2
        if VERBOSE:
            print(f"Checking page: {mid}")
        elif SHOW_PROGRESS:
            print_inline_status(f"Finding total pages... iteration {iteration}, checking page {mid}")
        current_signature = get_page_signature(mid)

        if current_signature == first_page_signature:
            high = mid - 1
        else:
            last_valid_page = mid
            low = mid + 1

    if SHOW_PROGRESS and not VERBOSE:
        print()
    return last_valid_page


def players_in_top_percent(total_players, fraction):
    """How many players are in the top `fraction` of the ladder; fractional counts round up."""
    if total_players <= 0:
        return 0
    return math.ceil(total_players * fraction)


def get_score_at_global_rank(global_rank):
    """
    Global rank is 1-based (rank 1 = highest points). Returns the score of that rank, or None if missing.
    """
    if global_rank < 1:
        return None
    page = (global_rank - 1) // 25 + 1
    idx = (global_rank - 1) % 25
    rows = get_hiscores_page(page)
    if idx >= len(rows):
        return None
    return rows[idx][1]


start_time = perf_counter()
total_steps = len(RANK_PERCENTILES) + 2
step = 0

total_pages = get_total_pages()
step += 1
if SHOW_PROGRESS:
    print_progress(step, total_steps, f"Total pages found: {total_pages}")
else:
    print(f"Total Pages: {total_pages}")

last_page_data = get_hiscores_page(total_pages)
total_players = last_page_data[-1][0] if last_page_data else 0
step += 1
if SHOW_PROGRESS:
    print_progress(step, total_steps, f"Total players found: {total_players:,}")
else:
    print(f"Total Players: {total_players}")

rank_results = {}
for rank_name, top_fraction in RANK_PERCENTILES:
    players_qualified = players_in_top_percent(total_players, top_fraction)
    point_cutoff = (
        get_score_at_global_rank(players_qualified) if players_qualified else None
    )
    rank_results[rank_name] = {
        "players_qualified": players_qualified,
        "point_cutoff": point_cutoff,
    }
    step += 1
    top_percent_label = f"{top_fraction * 100:g}%"
    cutoff_label = f"{point_cutoff:,}" if point_cutoff is not None else "N/A"
    if SHOW_PROGRESS:
        print_progress(
            step,
            total_steps,
            f"{rank_name} (top {top_percent_label}) -> {players_qualified:,} players, {cutoff_label} points",
        )
    else:
        print(
            f"{rank_name} top {top_percent_label}: {players_qualified} players, {cutoff_label} points"
        )

previous_data = load_previous_output()
previous_total_players = (
    previous_data.get("total_players")
    if previous_data and isinstance(previous_data.get("total_players"), int)
    else None
)

output_data = {
    "logged_at": datetime.now(timezone.utc).isoformat(),
    "total_players": total_players,
    "total_players_change": _change(total_players, previous_total_players),
    "ranks": [],
}

for rank_name, top_fraction in RANK_PERCENTILES:
    players_qualified = rank_results[rank_name]["players_qualified"]
    previous_players_qualified = _previous_rank_players(previous_data, rank_name)
    output_data["ranks"].append(
        {
            "name": rank_name,
            "point_cutoff": rank_results[rank_name]["point_cutoff"],
            "players_qualified": players_qualified,
            "players_qualified_change": _change(players_qualified, previous_players_qualified),
        }
    )

output_path = prepare_output_path()

with output_path.open("w", encoding="utf-8") as output_file:
    json.dump(output_data, output_file, indent=2)
    output_file.write("\n")

elapsed = perf_counter() - start_time
print()
print(f"Wrote rank data to {output_path}")
print(f"Completed in {elapsed:.2f}s")
