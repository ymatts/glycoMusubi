#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fetch_glyconnect_sites.py

Fetch site-specific glycosylation data linking (protein, site_position) -> glycan.

Strategy:
  1. Primary: GlyGen protein detail API (api.glygen.org/protein/detail/{uniprot_ac})
     - Returns glycosylation array with start_pos, residue, type, and glytoucan_ac
  2. Fallback: GlyConnect REST API (glyconnect.expasy.org)
  3. Fallback: GlyConnect SPARQL endpoint

Output: data_clean/glyconnect_site_glycans.tsv
  Columns: uniprot_ac, site_position, residue, glycosylation_type, glytoucan_ac

Usage:
  python3 scripts/fetch_glyconnect_sites.py
  python3 scripts/fetch_glyconnect_sites.py --limit 100  # test with 100 proteins
  python3 scripts/fetch_glyconnect_sites.py --source glygen  # force GlyGen only
"""

import os
import sys
import json
import time
import logging
import argparse
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_CLEAN_DIR = os.path.join(PROJECT_ROOT, "data_clean")
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data_raw")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
OUTPUT_PATH = os.path.join(DATA_CLEAN_DIR, "glyconnect_site_glycans.tsv")
CACHE_PATH = os.path.join(DATA_RAW_DIR, "glygen_protein_glycosylation_cache.json")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOGS_DIR, "fetch_glyconnect_sites.log")),
    ],
)
logger = logging.getLogger(__name__)


# ===========================================================================
# HTTP helpers (reuse patterns from scripts/utils/api_client.py)
# ===========================================================================

class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, requests_per_second: float = 5.0):
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()


def create_session(max_retries: int = 3, backoff_factor: float = 1.0) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_json(
    session: requests.Session,
    url: str,
    rate_limiter: RateLimiter,
    timeout: int = 60,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    method: str = "GET",
    **kwargs,
) -> Optional[Any]:
    """Fetch JSON with rate limiting + manual retry."""
    rate_limiter.wait()
    last_exc = None
    for attempt in range(max_retries):
        try:
            resp = session.request(method, url, timeout=timeout, **kwargs)
            if resp.status_code == 429:
                wait = backoff_factor ** attempt
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = max(wait, int(retry_after))
                    except ValueError:
                        pass
                logger.warning("Rate-limited (429) on %s; waiting %.1fs (attempt %d/%d)",
                               url, wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = backoff_factor ** attempt
                logger.warning("Server error %d on %s; waiting %.1fs (attempt %d/%d)",
                               resp.status_code, url, wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.JSONDecodeError:
            logger.error("JSON decode error for %s", url)
            return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            wait = backoff_factor ** attempt
            logger.warning("Network error for %s: %s; waiting %.1fs (attempt %d/%d)",
                           url, exc, wait, attempt + 1, max_retries)
            time.sleep(wait)
        except requests.exceptions.HTTPError as exc:
            last_exc = exc
            if exc.response is not None and exc.response.status_code < 500:
                logger.error("HTTP %d for %s", exc.response.status_code, url)
                return None
            wait = backoff_factor ** attempt
            time.sleep(wait)

    logger.error("Failed after %d retries: %s (last error: %s)", max_retries, url, last_exc)
    return None


# ===========================================================================
# Load proteins from existing KG data
# ===========================================================================

def load_protein_ids() -> List[str]:
    """Load unique UniProt accession IDs from proteins_clean.tsv and enzymes_clean.tsv.

    Returns both isoform-suffixed IDs (e.g. P01588-1) and canonical IDs (P01588).
    The GlyGen API accepts both formats.  We deduplicate by canonical accession
    and prefer the isoform-suffixed form from the data file so that output IDs
    are consistent with the rest of the KG.
    """
    # Map canonical -> isoform-suffixed ID (keep first encountered)
    canonical_to_full: Dict[str, str] = {}

    proteins_path = os.path.join(DATA_CLEAN_DIR, "proteins_clean.tsv")
    enzymes_path = os.path.join(DATA_CLEAN_DIR, "enzymes_clean.tsv")

    for path, col in [(proteins_path, "protein_id"), (enzymes_path, "enzyme_id")]:
        if not os.path.exists(path):
            logger.warning("File not found: %s", path)
            continue
        with open(path) as fh:
            header = fh.readline().strip().split("\t")
            idx = header.index(col) if col in header else 0
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) > idx:
                    uid = parts[idx].strip()
                    if not uid:
                        continue
                    canonical = uid.split("-")[0] if "-" in uid else uid
                    if canonical not in canonical_to_full:
                        canonical_to_full[canonical] = uid

    result = sorted(canonical_to_full.values())
    logger.info("Loaded %d unique UniProt IDs from KG data", len(result))
    return result


def load_known_glytoucan_ids() -> Set[str]:
    """Load GlyTouCan IDs already in the KG for filtering."""
    glycans_path = os.path.join(DATA_CLEAN_DIR, "glycans_clean.tsv")
    ids: Set[str] = set()
    if os.path.exists(glycans_path):
        with open(glycans_path) as fh:
            header = fh.readline().strip().split("\t")
            idx = header.index("glycan_id") if "glycan_id" in header else 0
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) > idx:
                    ids.add(parts[idx].strip())
    logger.info("Loaded %d GlyTouCan IDs from existing KG", len(ids))
    return ids


# ===========================================================================
# Strategy 1: GlyGen protein detail API
# ===========================================================================

def fetch_glygen_protein_glycosylation(
    protein_ids: List[str],
    session: requests.Session,
    rate_limiter: RateLimiter,
    cache: Optional[Dict] = None,
    cache_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch glycosylation data per-protein from the GlyGen protein detail API.

    Endpoint: GET https://api.glygen.org/protein/detail/<uniprot_ac>

    The response includes a 'glycosylation' array where each entry has:
      - type (N-linked, O-linked, etc.)
      - glytoucan_ac (GlyTouCan accession)
      - start_pos / end_pos
      - start_aa (residue, e.g. "Asn")
      - residue (residue, e.g. "Asn")
      - site_lbl (site label, e.g. "Asn110")
      - site_category (e.g. "reported_with_glycan", "reported")
      - evidence (list of evidence objects)

    Args:
        protein_ids: List of UniProt accession IDs (may include isoform suffix).
        session: HTTP session.
        rate_limiter: Rate limiter.
        cache: Optional dict for caching responses.
        cache_path: If given, save cache incrementally every 500 proteins.

    Returns list of dicts with keys:
      uniprot_ac, site_position, residue, glycosylation_type, glytoucan_ac
    """
    if cache is None:
        cache = {}

    records: List[Dict[str, Any]] = []
    success_count = 0
    fail_count = 0
    skip_count = 0

    total = len(protein_ids)
    logger.info("Fetching glycosylation from GlyGen protein detail API for %d proteins...", total)

    for i, uid in enumerate(protein_ids):
        if (i + 1) % 200 == 0 or i == 0:
            logger.info("  Progress: %d/%d (success=%d, fail=%d, skip=%d, records=%d)",
                        i + 1, total, success_count, fail_count, skip_count, len(records))

        # Incremental cache save every 500 proteins
        if cache_path and (i + 1) % 500 == 0:
            save_cache(cache, cache_path)

        # Normalise: the API accepts both P01588 and P01588-1
        # Cache key is the canonical form to avoid duplicate fetches
        canonical = uid.split("-")[0] if "-" in uid else uid

        # Check cache (keyed by canonical ID)
        if canonical in cache:
            glyc_data = cache[canonical]
            if glyc_data is not None:
                records.extend(glyc_data)
                success_count += 1
            else:
                skip_count += 1
            continue

        # Use GET endpoint (POST returns 404 for this API)
        url = f"https://api.glygen.org/protein/detail/{uid}"
        data = _fetch_glygen_protein_detail(session, url, rate_limiter)

        # If isoform-suffixed ID fails, try canonical
        if data is None and "-" in uid:
            url = f"https://api.glygen.org/protein/detail/{canonical}"
            data = _fetch_glygen_protein_detail(session, url, rate_limiter)

        if data is None or not isinstance(data, dict):
            fail_count += 1
            cache[canonical] = None
            continue

        # Check for error response
        if "error" in data or "error_list" in data:
            fail_count += 1
            cache[canonical] = None
            continue

        # Extract glycosylation data
        glyc_list = data.get("glycosylation", [])
        if not glyc_list:
            glyc_list = data.get("site_annotation", [])
        if not glyc_list:
            gp = data.get("glycoprotein", {})
            if isinstance(gp, dict):
                glyc_list = gp.get("glycosylation", [])

        protein_records = _parse_glygen_glycosylation_entries(canonical, glyc_list)

        cache[canonical] = protein_records if protein_records else None
        records.extend(protein_records)

        if protein_records:
            success_count += 1
        else:
            skip_count += 1

    logger.info("GlyGen fetch complete: %d proteins with data, %d failed, %d skipped (no glycosylation)",
                success_count, fail_count, skip_count)
    logger.info("Total records extracted: %d", len(records))
    return records


def _fetch_glygen_protein_detail(
    session: requests.Session, url: str, rate_limiter: RateLimiter
) -> Optional[Dict]:
    """Fetch a single protein detail with a streaming size guard.

    Some proteins (e.g. heavily glycosylated mucins) return multi-MB responses.
    We stream the response and abort if it exceeds 10 MB.
    """
    MAX_BYTES = 10 * 1024 * 1024  # 10 MB
    rate_limiter.wait()

    for attempt in range(3):
        try:
            resp = session.get(url, timeout=60, stream=True, headers={"Accept": "application/json"})
            if resp.status_code == 404:
                resp.close()
                return None
            if resp.status_code == 429 or resp.status_code >= 500:
                resp.close()
                wait = 2.0 ** attempt
                time.sleep(wait)
                continue
            resp.raise_for_status()

            # Read with size limit
            chunks = []
            total_size = 0
            for chunk in resp.iter_content(chunk_size=65536):
                total_size += len(chunk)
                if total_size > MAX_BYTES:
                    logger.warning("Response too large (>%d MB) for %s; skipping",
                                   MAX_BYTES // (1024*1024), url)
                    resp.close()
                    return None
                chunks.append(chunk)
            resp.close()

            body = b"".join(chunks)
            return json.loads(body)

        except requests.exceptions.JSONDecodeError:
            logger.error("JSON decode error for %s", url)
            return None
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            wait = 2.0 ** attempt
            logger.warning("Network error for %s: %s; retry in %.1fs", url, exc, wait)
            time.sleep(wait)
        except Exception as exc:
            logger.error("Unexpected error for %s: %s", url, exc)
            return None

    logger.error("Failed after 3 retries: %s", url)
    return None


def _parse_glygen_glycosylation_entries(
    uniprot_ac: str, glyc_list: List[Dict]
) -> List[Dict[str, Any]]:
    """Parse the 'glycosylation' array from GlyGen protein detail response."""
    protein_records = []
    for entry in glyc_list:
        if not isinstance(entry, dict):
            continue

        # Extract GlyTouCan accession
        glytoucan_ac = entry.get("glytoucan_ac", "")
        if not glytoucan_ac:
            glycan_obj = entry.get("glycan", {})
            if isinstance(glycan_obj, dict):
                glytoucan_ac = glycan_obj.get("glytoucan_ac", "")
        if not glytoucan_ac:
            continue  # Skip entries without a glycan identifier

        # Extract site position
        site_pos = entry.get("start_pos")
        if site_pos is None:
            site_pos = entry.get("position")
        if site_pos is None:
            site_lbl = entry.get("site_lbl", "")
            # Parse position from site_lbl like "Asn110"
            if isinstance(site_lbl, str):
                import re
                m = re.search(r"(\d+)$", site_lbl)
                if m:
                    site_pos = int(m.group(1))
        if site_pos is None:
            continue

        try:
            site_pos = int(site_pos)
        except (ValueError, TypeError):
            continue

        # Extract residue (3-letter or 1-letter)
        residue = entry.get("start_aa", "") or entry.get("residue", "")
        if not residue:
            site_lbl = entry.get("site_lbl", "")
            if isinstance(site_lbl, str) and len(site_lbl) >= 1 and site_lbl[0].isalpha():
                residue = site_lbl[0]

        # Extract glycosylation type
        glyc_type = entry.get("type", "") or entry.get("glycosylation_type", "")
        if not glyc_type:
            glyc_type = entry.get("site_category", "")

        protein_records.append({
            "uniprot_ac": uniprot_ac,
            "site_position": site_pos,
            "residue": residue,
            "glycosylation_type": glyc_type,
            "glytoucan_ac": glytoucan_ac,
        })

    return protein_records


# ===========================================================================
# Strategy 2: GlyConnect REST API
# ===========================================================================

def fetch_glyconnect_glycosylations(
    session: requests.Session,
    rate_limiter: RateLimiter,
    tax_id: int = 9606,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch glycosylation records from GlyConnect REST API.

    Endpoint: GET https://glyconnect.expasy.org/api/glycosylations?tax_id=9606
    Supports pagination via offset/limit or page parameters.

    Returns list of dicts with keys:
      uniprot_ac, site_position, residue, glycosylation_type, glytoucan_ac
    """
    records: List[Dict[str, Any]] = []

    # Try paginated fetch
    base_url = "https://glyconnect.expasy.org/api/glycosylations"
    page_size = 1000
    offset = 0

    logger.info("Fetching glycosylation data from GlyConnect API (tax_id=%d)...", tax_id)

    # First, try a small request to see what the API returns
    test_url = f"{base_url}?taxid={tax_id}&limit=5"
    test_data = fetch_json(session, test_url, rate_limiter, timeout=60)

    if test_data is None:
        # Try alternative parameter names
        test_url = f"{base_url}?tax_id={tax_id}&limit=5"
        test_data = fetch_json(session, test_url, rate_limiter, timeout=60)

    if test_data is None:
        # Try without pagination
        test_url = f"{base_url}?taxid={tax_id}"
        test_data = fetch_json(session, test_url, rate_limiter, timeout=120)

    if test_data is None:
        # Try the proteins endpoint instead
        test_url = "https://glyconnect.expasy.org/api/proteins?taxid=9606&limit=5"
        test_data = fetch_json(session, test_url, rate_limiter, timeout=60)

    if test_data is None:
        logger.warning("GlyConnect API not responding. Skipping this source.")
        return records

    logger.info("GlyConnect test response type: %s", type(test_data).__name__)

    # Parse the response structure
    if isinstance(test_data, list):
        # Direct list of records
        all_data = test_data if limit is None else test_data[:limit]
        records = _parse_glyconnect_records(all_data)

        # If it returned few records, try paginating
        if len(test_data) >= 5:
            logger.info("GlyConnect returned list of %d items; attempting full fetch...", len(test_data))
            full_url = f"{base_url}?taxid={tax_id}"
            if limit:
                full_url += f"&limit={limit}"
            full_data = fetch_json(session, full_url, rate_limiter, timeout=300)
            if full_data and isinstance(full_data, list):
                records = _parse_glyconnect_records(full_data)

    elif isinstance(test_data, dict):
        # Paginated response with results/total/count keys
        results_key = None
        for key in ["results", "data", "glycosylations", "items", "entries"]:
            if key in test_data:
                results_key = key
                break

        if results_key:
            total = test_data.get("total", test_data.get("count", 0))
            logger.info("GlyConnect paginated: total=%s, results_key=%s", total, results_key)

            # Fetch all pages
            while True:
                page_url = f"{base_url}?taxid={tax_id}&offset={offset}&limit={page_size}"
                page_data = fetch_json(session, page_url, rate_limiter, timeout=120)

                if page_data is None:
                    break
                if isinstance(page_data, dict):
                    page_items = page_data.get(results_key, [])
                elif isinstance(page_data, list):
                    page_items = page_data
                else:
                    break

                if not page_items:
                    break

                records.extend(_parse_glyconnect_records(page_items))
                offset += len(page_items)

                if len(page_items) < page_size:
                    break
                if limit and len(records) >= limit:
                    break
        else:
            # Single object — try to parse directly
            records = _parse_glyconnect_records([test_data])

    logger.info("GlyConnect fetch complete: %d records", len(records))
    return records


def _parse_glyconnect_records(items: List[Dict]) -> List[Dict[str, Any]]:
    """Parse GlyConnect glycosylation records into uniform format."""
    records = []
    for item in items:
        if not isinstance(item, dict):
            continue

        # GlyConnect may have different field names
        uniprot_ac = (
            item.get("uniprot_ac")
            or item.get("protein", {}).get("uniprot_ac", "")
            or item.get("uniprotAc", "")
            or item.get("protein_acc", "")
        )
        if isinstance(item.get("protein"), dict):
            uniprot_ac = uniprot_ac or item["protein"].get("uniprotkb_ac", "")

        site_pos = (
            item.get("site_position")
            or item.get("position")
            or item.get("start_pos")
        )
        if isinstance(item.get("site"), dict):
            site_pos = site_pos or item["site"].get("position", "")

        residue = (
            item.get("residue")
            or item.get("amino_acid")
            or item.get("start_aa", "")
        )

        glyc_type = (
            item.get("glycosylation_type")
            or item.get("type")
            or item.get("category", "")
        )

        glytoucan_ac = (
            item.get("glytoucan_ac")
            or item.get("glytoucan_id")
            or item.get("glycan_id", "")
        )
        if isinstance(item.get("structure"), dict):
            glytoucan_ac = glytoucan_ac or item["structure"].get("glytoucan_ac", "")
        if isinstance(item.get("glycan"), dict):
            glytoucan_ac = glytoucan_ac or item["glycan"].get("glytoucan_ac", "")

        if not uniprot_ac or not glytoucan_ac:
            continue

        try:
            site_pos = int(site_pos) if site_pos else None
        except (ValueError, TypeError):
            site_pos = None

        if site_pos is None:
            continue

        records.append({
            "uniprot_ac": uniprot_ac,
            "site_position": site_pos,
            "residue": residue or "",
            "glycosylation_type": glyc_type or "",
            "glytoucan_ac": glytoucan_ac,
        })

    return records


# ===========================================================================
# Strategy 3: GlyConnect SPARQL fallback
# ===========================================================================

def fetch_glyconnect_sparql(
    session: requests.Session,
    rate_limiter: RateLimiter,
) -> List[Dict[str, Any]]:
    """
    Fetch site-glycan mappings from GlyConnect SPARQL endpoint.

    Endpoint: https://glyconnect.expasy.org/sparql
    """
    sparql_url = "https://glyconnect.expasy.org/sparql"

    query = """
    PREFIX glyconnect: <https://purl.org/glyconnect/>
    PREFIX up: <http://purl.uniprot.org/core/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT ?protein ?position ?residue ?glycotype ?glytoucan
    WHERE {
      ?glycosylation a glyconnect:Glycosylation ;
                     glyconnect:protein ?proteinNode ;
                     glyconnect:position ?position ;
                     glyconnect:structure ?structureNode .

      ?proteinNode glyconnect:uniprotkb ?protein .
      ?structureNode glyconnect:glytoucan ?glytoucan .

      OPTIONAL { ?glycosylation glyconnect:amino_acid ?residue . }
      OPTIONAL { ?glycosylation glyconnect:type ?glycotype . }

      # Human proteins only (taxid 9606)
      ?proteinNode glyconnect:taxonomy <http://purl.uniprot.org/taxonomy/9606> .
    }
    """

    logger.info("Attempting GlyConnect SPARQL query...")
    rate_limiter.wait()

    try:
        resp = session.get(
            sparql_url,
            params={"query": query, "format": "json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=120,
        )
        if resp.status_code != 200:
            logger.warning("GlyConnect SPARQL returned %d", resp.status_code)
            return []

        data = resp.json()
        bindings = data.get("results", {}).get("bindings", [])

        records = []
        for b in bindings:
            uniprot_ac = b.get("protein", {}).get("value", "")
            # Extract just the accession from URI if needed
            if "/" in uniprot_ac:
                uniprot_ac = uniprot_ac.rsplit("/", 1)[-1]

            site_pos = b.get("position", {}).get("value", "")
            residue = b.get("residue", {}).get("value", "")
            glyc_type = b.get("glycotype", {}).get("value", "")
            glytoucan_ac = b.get("glytoucan", {}).get("value", "")
            if "/" in glytoucan_ac:
                glytoucan_ac = glytoucan_ac.rsplit("/", 1)[-1]

            if not uniprot_ac or not glytoucan_ac:
                continue

            try:
                site_pos = int(site_pos)
            except (ValueError, TypeError):
                continue

            records.append({
                "uniprot_ac": uniprot_ac,
                "site_position": site_pos,
                "residue": residue,
                "glycosylation_type": glyc_type,
                "glytoucan_ac": glytoucan_ac,
            })

        logger.info("GlyConnect SPARQL returned %d records", len(records))
        return records

    except Exception as exc:
        logger.warning("GlyConnect SPARQL failed: %s", exc)
        return []


# ===========================================================================
# Deduplication and output
# ===========================================================================

def deduplicate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate by (uniprot_ac, site_position, glytoucan_ac) tuple."""
    seen: Set[Tuple[str, int, str]] = set()
    unique = []
    for rec in records:
        key = (rec["uniprot_ac"], rec["site_position"], rec["glytoucan_ac"])
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique


def save_tsv(records: List[Dict[str, Any]], output_path: str):
    """Save records to TSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    columns = ["uniprot_ac", "site_position", "residue", "glycosylation_type", "glytoucan_ac"]

    with open(output_path, "w") as fh:
        fh.write("\t".join(columns) + "\n")
        for rec in records:
            row = [str(rec.get(c, "")) for c in columns]
            fh.write("\t".join(row) + "\n")

    logger.info("Saved %d records to %s", len(records), output_path)


def save_cache(cache: Dict, path: str):
    """Save API response cache to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert None values to null-safe form
    safe = {}
    for k, v in cache.items():
        safe[k] = v if v is not None else []
    with open(path, "w") as fh:
        json.dump(safe, fh)
    logger.info("Saved cache (%d entries) to %s", len(cache), path)


def load_cache(path: str) -> Dict:
    """Load API response cache from JSON if it exists."""
    if os.path.exists(path):
        try:
            with open(path) as fh:
                data = json.load(fh)
            logger.info("Loaded cache with %d entries from %s", len(data), path)
            # Convert empty lists back to None for skip detection
            for k in data:
                if data[k] == []:
                    data[k] = None
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load cache: %s", exc)
    return {}


def report_statistics(records: List[Dict[str, Any]], known_glycans: Set[str]):
    """Print comprehensive statistics."""
    proteins = set(r["uniprot_ac"] for r in records)
    sites = set((r["uniprot_ac"], r["site_position"]) for r in records)
    glycans = set(r["glytoucan_ac"] for r in records)
    triples = set((r["uniprot_ac"], r["site_position"], r["glytoucan_ac"]) for r in records)
    glyc_types = defaultdict(int)
    for r in records:
        glyc_types[r.get("glycosylation_type", "unknown") or "unknown"] += 1

    overlap = glycans & known_glycans

    logger.info("=" * 60)
    logger.info("STATISTICS")
    logger.info("=" * 60)
    logger.info("Unique proteins:        %d", len(proteins))
    logger.info("Unique (protein, site):  %d", len(sites))
    logger.info("Unique glycans:          %d", len(glycans))
    logger.info("Unique triples:          %d", len(triples))
    logger.info("Glycans overlapping KG:  %d / %d (%.1f%%)",
                len(overlap), len(glycans),
                100.0 * len(overlap) / len(glycans) if glycans else 0)
    logger.info("Glycosylation types:")
    for gtype, count in sorted(glyc_types.items(), key=lambda x: -x[1]):
        logger.info("  %-25s %d", gtype, count)
    logger.info("=" * 60)

    # Also print to stdout for visibility
    print(f"\n{'='*60}")
    print("SITE-GLYCAN DATA STATISTICS")
    print(f"{'='*60}")
    print(f"Unique proteins:        {len(proteins)}")
    print(f"Unique (protein, site): {len(sites)}")
    print(f"Unique glycans:         {len(glycans)}")
    print(f"Unique triples:         {len(triples)}")
    print(f"Glycans overlapping KG: {len(overlap)} / {len(glycans)} "
          f"({100.0*len(overlap)/len(glycans):.1f}%)" if glycans else "N/A")
    print(f"\nGlycosylation types:")
    for gtype, count in sorted(glyc_types.items(), key=lambda x: -x[1]):
        print(f"  {gtype:<25s} {count}")
    print(f"{'='*60}\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fetch site-specific glycosylation data for site-level glycan prediction."
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of proteins to query (for testing)")
    parser.add_argument("--source", choices=["glygen", "glyconnect", "sparql", "all"],
                        default="all",
                        help="Data source to use (default: all)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cached API responses")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH,
                        help="Output TSV path")
    parser.add_argument("--rate-limit", type=float, default=5.0,
                        help="Max requests per second (default: 5)")
    args = parser.parse_args()

    logger.info("Starting site-glycan data fetch")
    logger.info("Source: %s, Limit: %s, Rate limit: %.1f req/s",
                args.source, args.limit or "all", args.rate_limit)

    # Load protein IDs from existing KG
    protein_ids = load_protein_ids()
    if args.limit:
        protein_ids = protein_ids[:args.limit]
        logger.info("Limited to %d proteins", len(protein_ids))

    known_glycans = load_known_glytoucan_ids()

    # Setup HTTP
    session = create_session(max_retries=3, backoff_factor=1.0)
    rate_limiter = RateLimiter(args.rate_limit)

    all_records: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Strategy 1: GlyGen protein detail API
    # -----------------------------------------------------------------------
    if args.source in ("glygen", "all"):
        cache = {} if args.no_cache else load_cache(CACHE_PATH)
        glygen_records = fetch_glygen_protein_glycosylation(
            protein_ids, session, rate_limiter, cache=cache,
            cache_path=CACHE_PATH,
        )
        all_records.extend(glygen_records)
        logger.info("GlyGen contributed %d records", len(glygen_records))

        # Save cache for resume
        save_cache(cache, CACHE_PATH)

    # -----------------------------------------------------------------------
    # Strategy 2: GlyConnect REST API
    # -----------------------------------------------------------------------
    if args.source in ("glyconnect", "all"):
        gc_records = fetch_glyconnect_glycosylations(session, rate_limiter)
        all_records.extend(gc_records)
        logger.info("GlyConnect API contributed %d records", len(gc_records))

    # -----------------------------------------------------------------------
    # Strategy 3: GlyConnect SPARQL
    # -----------------------------------------------------------------------
    if args.source in ("sparql", "all"):
        sparql_records = fetch_glyconnect_sparql(session, rate_limiter)
        all_records.extend(sparql_records)
        logger.info("GlyConnect SPARQL contributed %d records", len(sparql_records))

    session.close()

    # -----------------------------------------------------------------------
    # Deduplicate and save
    # -----------------------------------------------------------------------
    if not all_records:
        logger.error("No records fetched from any source. Check network connectivity and API status.")
        sys.exit(1)

    unique_records = deduplicate_records(all_records)
    logger.info("After dedup: %d unique records (from %d raw)", len(unique_records), len(all_records))

    save_tsv(unique_records, args.output)
    report_statistics(unique_records, known_glycans)

    logger.info("Done. Output: %s", args.output)


if __name__ == "__main__":
    main()
