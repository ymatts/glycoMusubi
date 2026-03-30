#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
download_data.py

Downloads data from GlyGen, GlyTouCan, UniProt, and ChEMBL APIs.
Includes proper error handling, retry logic, and configurable limits.
"""

import os
import sys
import argparse
import pandas as pd
import time
from tqdm import tqdm
import json
import logging
import random
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from utils.config_loader import load_config, get_config
from utils.api_client import APIClient, APIFetchResult, create_session_with_retry
from utils.glytoucan_parser import GlyTouCanParser, extract_wurcs_from_jsonld
from utils.parallel import (
    get_worker_count, chunked, seed_worker,
    run_in_process_pool, run_sequential
)

config = load_config()


def get_parallel_settings():
    """Get parallel processing settings from config and environment."""
    import os
    
    # Check environment variables first (set by pipeline.py)
    parallel_env = os.environ.get('GLYCO_KG_PARALLEL', '').lower()
    if parallel_env == 'true':
        parallel_enabled = True
    elif parallel_env == 'false':
        parallel_enabled = False
    else:
        parallel_enabled = config.parallel.enabled
    
    workers_env = os.environ.get('GLYCO_KG_WORKERS')
    if workers_env:
        workers = int(workers_env)
    else:
        workers = get_worker_count(None, config.parallel.workers)
    
    seed_env = os.environ.get('GLYCO_KG_PARALLEL_SEED')
    if seed_env:
        seed = int(seed_env)
    else:
        seed = config.parallel.seed
    
    batch_size_env = os.environ.get('GLYCO_KG_BATCH_SIZE')
    if batch_size_env:
        batch_size = int(batch_size_env)
    else:
        batch_size = config.parallel.batch_size
    
    return parallel_enabled, workers, seed, batch_size


# Worker functions for multiprocessing (must be top-level for pickling)

def _process_glytoucan_batch(args):
    """
    Worker function to process a batch of GlyTouCan IDs using SPARQList API.
    
    Uses the new GlyCosmos SPARQList gtcid2seqs endpoint:
    GET https://api.glycosmos.org/sparqlist/gtcid2seqs?gtcid={id}
    
    Args:
        args: Tuple of (glycan_ids, config_dict, worker_id)
    
    Returns:
        Dict with results, failures, and parser statistics.
    """
    import time
    
    glycan_ids, config_dict, worker_id = args
    seed_worker(worker_id, config_dict.get('seed', 42))
    
    from utils.api_client import APIClient
    from utils.glytoucan_parser import GlyTouCanParser
    
    results = []
    failures = []
    parser = GlyTouCanParser()
    
    # New SPARQList endpoint
    base_url = config_dict.get('base_url', 'https://api.glycosmos.org/sparqlist')
    endpoint = config_dict.get('gtcid2seqs_endpoint', 'gtcid2seqs')
    min_interval_ms = config_dict.get('min_request_interval_ms', 200)
    min_interval_sec = min_interval_ms / 1000.0
    
    last_request_time = 0.0
    
    with APIClient(
        max_retries=config_dict.get('max_retries', 3),
        backoff_factor=config_dict.get('backoff_seconds', 2),
        timeout=config_dict.get('timeout', 30),
        rate_limit=config_dict.get('rate_limit', 5)
    ) as client:
        for glycan_id in glycan_ids:
            # Rate limiting: ensure minimum interval between requests
            current_time = time.time()
            elapsed = current_time - last_request_time
            if elapsed < min_interval_sec and last_request_time > 0:
                time.sleep(min_interval_sec - elapsed)
            
            # New SPARQList URL format: /sparqlist/gtcid2seqs?gtcid={id}
            url = f"{base_url}/{endpoint}?gtcid={glycan_id}"
            
            try:
                data = client.get_json(url)
                last_request_time = time.time()
                
                if data is None:
                    failures.append({
                        'id': glycan_id,
                        'error_type': 'no_response',
                        'message': 'No response or invalid JSON'
                    })
                    continue
                
                # Use the new SPARQList parser
                wurcs = parser.parse_sparqlist_response(data, glycan_id)
                
                if wurcs:
                    results.append({'glycan_id': glycan_id, 'structure': wurcs})
                else:
                    failures.append({
                        'id': glycan_id,
                        'error_type': 'no_wurcs',
                        'message': 'WURCS not found in SPARQList response'
                    })
                    
            except Exception as e:
                failures.append({
                    'id': glycan_id,
                    'error_type': 'exception',
                    'message': str(e)
                })
    
    return {
        'results': results,
        'failures': failures,
        'parser_stats': parser.get_statistics()
    }


def _process_uniprot_batch(args):
    """
    Worker function to process a batch of UniProt IDs.
    
    Args:
        args: Tuple of (uniprot_ids, config_dict, worker_id)
    
    Returns:
        Dict with results and failures.
    """
    uniprot_ids, config_dict, worker_id = args
    seed_worker(worker_id, config_dict.get('seed', 42))
    
    from utils.api_client import APIClient
    
    results = []
    failures = []
    
    base_url = config_dict.get('base_url', 'https://rest.uniprot.org/uniprotkb')
    
    with APIClient(
        max_retries=config_dict.get('max_retries', 3),
        backoff_factor=config_dict.get('backoff_seconds', 2),
        timeout=config_dict.get('timeout', 30),
        rate_limit=config_dict.get('rate_limit', 10)
    ) as client:
        for uid in uniprot_ids:
            url = f"{base_url}/{uid}.json"
            
            try:
                data = client.get_json(url)
                
                if data is None:
                    failures.append({
                        'id': uid,
                        'error_type': 'no_response',
                        'message': 'No response or invalid JSON'
                    })
                    continue
                
                diseases = []
                comments = data.get('comments', [])
                for comment in comments:
                    if comment.get('commentType') == 'DISEASE':
                        disease_info = comment.get('disease', {})
                        disease_id = disease_info.get('diseaseId')
                        disease_name_obj = disease_info.get('diseaseName', {})
                        disease_name = disease_name_obj.get('value') if isinstance(disease_name_obj, dict) else disease_name_obj
                        if disease_id:
                            diseases.append(f"{disease_id}|{disease_name or ''}")

                variants = []
                features = data.get('features', [])
                for feature in features:
                    if feature.get('type') == 'VARIANT':
                        var_id = feature.get('featureId')
                        desc = feature.get('description', '')
                        if var_id:
                            variants.append(f"{var_id}|{desc}")

                # Subcellular locations
                subcellular_locations = []
                for comment in comments:
                    if comment.get('commentType') == 'SUBCELLULAR LOCATION':
                        for sub_loc in comment.get('subcellularLocations', []):
                            loc_value = sub_loc.get('location', {})
                            if isinstance(loc_value, dict):
                                raw_loc = loc_value.get('value', '')
                            else:
                                raw_loc = str(loc_value)
                            if raw_loc:
                                subcellular_locations.append(raw_loc)

                gene_name = ""
                if 'genes' in data and data['genes']:
                    gene_name_obj = data['genes'][0].get('geneName', {})
                    gene_name = gene_name_obj.get('value', '') if isinstance(gene_name_obj, dict) else str(gene_name_obj)
                
                sequence = ""
                seq_obj = data.get('sequence', {})
                if isinstance(seq_obj, dict):
                    sequence = str(seq_obj.get('value', '') or '').strip()

                results.append({
                    'uniprot_id': uid,
                    'diseases': ";".join(diseases),
                    'variants': ";".join(variants),
                    'subcellular_locations': ";".join(subcellular_locations),
                    'gene_symbol': gene_name,
                    'sequence': sequence
                })

            except Exception as e:
                failures.append({
                    'id': uid,
                    'error_type': 'exception',
                    'message': str(e)
                })
    
    return {
        'results': results,
        'failures': failures
    }


def _process_glygen_glycan_detail_batch(args):
    """
    Worker function to process a batch of glycan IDs for GlyGen glycan detail API.
    
    This fetches detailed glycan information including:
    - enzyme: enzymes that produce/consume the glycan
    - associated_proteins: proteins associated with the glycan (replaces deprecated usecases API)
    - subsumption: glycan hierarchy relationships
    - motifs: glycan motif structures
    - residues: glycan residue information
    - classification: glycan classification
    
    The GlyGen detail API (https://api.glygen.org/glycan/detail/{id}) is the ONLY valid
    live API for GlyGen data. The usecases API (/usecases/glycan_to_glycoproteins/) is
    deprecated and always returns empty results.
    
    Args:
        args: Tuple of (glycan_ids, config_dict, worker_id)
    
    Returns:
        Dict with glycan_details and failures.
    """
    glycan_ids, config_dict, worker_id = args
    seed_worker(worker_id, config_dict.get('seed', 42))
    
    from utils.api_client import APIClient
    
    glycan_details = []
    failures = []
    
    with APIClient(
        max_retries=config_dict.get('max_retries', 3),
        backoff_factor=config_dict.get('backoff_seconds', 2),
        timeout=config_dict.get('timeout', 30),
        rate_limit=config_dict.get('rate_limit', 10)
    ) as client:
        for glycan_id in glycan_ids:
            url = f"https://api.glygen.org/glycan/detail/{glycan_id}"
            
            try:
                data = client.get_json(url)
                
                if data is None:
                    failures.append({
                        'id': glycan_id,
                        'error_type': 'no_response',
                        'message': 'No response or invalid JSON'
                    })
                    continue
                
                if isinstance(data, dict):
                    if 'error' in data:
                        failures.append({
                            'id': glycan_id,
                            'error_type': 'api_error',
                            'message': data.get('error', 'Unknown error')
                        })
                        continue
                    
                    # Check for empty response (API returned but no meaningful data)
                    glytoucan_data = data.get('glytoucan', {})
                    if not glytoucan_data or not glytoucan_data.get('glytoucan_ac'):
                        failures.append({
                            'id': glycan_id,
                            'error_type': 'empty_response',
                            'message': 'Empty or invalid glytoucan data in response'
                        })
                        continue
                    
                    # Extract associated proteins with key-fallback handling.
                    associated_proteins, assoc_source_key = _extract_associated_proteins(data)

                    # Extract relevant data including associated_proteins.
                    # associated_proteins replaces the deprecated usecases API.
                    detail = {
                        'glycan_id': glycan_id,
                        'glytoucan': glytoucan_data.get('glytoucan_ac', glycan_id),
                        'mass': data.get('mass'),
                        'glycan_type': data.get('glycan_type'),
                        'enzymes': data.get('enzyme', []),
                        'associated_proteins': associated_proteins,
                        'associated_proteins_source_key': assoc_source_key,
                        'subsumption': data.get('subsumption', []),
                        'motifs': data.get('motifs', []),
                        'residues': data.get('residues', []),
                        'classification': data.get('classification', []),
                        'species': data.get('species', []),
                    }
                    glycan_details.append(detail)
                else:
                    failures.append({
                        'id': glycan_id,
                        'error_type': 'unexpected_format',
                        'message': f'Expected dict, got {type(data).__name__}'
                    })
                        
            except Exception as e:
                failures.append({
                    'id': glycan_id,
                    'error_type': 'exception',
                    'message': str(e)
                })
    
    return {
        'glycan_details': glycan_details,
        'failures': failures
    }


def _normalize_associated_protein_entry(entry):
    """
    Normalize one associated-protein record to a common shape.
    Returns dict with at least 'uniprot_canonical_ac' when resolvable.
    """
    if isinstance(entry, str):
        uid = entry.strip()
        if uid:
            return {'uniprot_canonical_ac': uid}
        return None

    if not isinstance(entry, dict):
        return None

    uid = (
        entry.get('uniprot_canonical_ac')
        or entry.get('uniprot_id')
        or entry.get('uniprot')
        or entry.get('accession')
        or entry.get('primary_accession')
        or entry.get('protein_id')
    )
    uid = str(uid).strip() if uid is not None else ""
    if not uid:
        return None

    gene = entry.get('gene') or entry.get('gene_symbol') or ""
    pname = entry.get('protein_name') or entry.get('name') or ""

    return {
        'uniprot_canonical_ac': uid,
        'gene': gene,
        'protein_name': pname,
    }


def _extract_associated_proteins(data):
    """
    Extract associated proteins from GlyGen detail response with fallback keys.
    Returns (normalized_list, source_key).
    """
    if not isinstance(data, dict):
        return [], "none"

    candidate_keys = [
        'glycoprotein',
        'associated_proteins',
        'associatedProteins',
        'glycoproteins',
        'proteins',
    ]

    for key in candidate_keys:
        value = data.get(key, [])
        if not isinstance(value, list):
            continue
        normalized = []
        for entry in value:
            n = _normalize_associated_protein_entry(entry)
            if n:
                normalized.append(n)
        if normalized:
            return normalized, key
        # Keep the first matched key even if empty for diagnostics.
        if key in data:
            return [], key

    # Some APIs may use nested payloads.
    nested = data.get('glygen')
    if isinstance(nested, dict):
        for key in candidate_keys:
            value = nested.get(key, [])
            if not isinstance(value, list):
                continue
            normalized = []
            for entry in value:
                n = _normalize_associated_protein_entry(entry)
                if n:
                    normalized.append(n)
            if normalized:
                return normalized, f"glygen.{key}"
            if key in nested:
                return [], f"glygen.{key}"

    return [], "none"


DATA_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", config.directories.raw_data)
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", config.directories.logs)
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Minimum file size for valid cache (bytes)
MIN_VALID_CACHE_BYTES = 50


def is_valid_cache(path: str, min_bytes: int = MIN_VALID_CACHE_BYTES) -> bool:
    """
    Check if a cached file is valid (exists and has sufficient size).
    
    Args:
        path: Path to the cached file.
        min_bytes: Minimum file size in bytes to be considered valid.
    
    Returns:
        True if the file exists and is large enough, False otherwise.
    """
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    if size < min_bytes:
        logger.warning(f"{path} appears empty or corrupted (size={size} bytes). Ignoring cache.")
        return False
    return True


def normalize_uniprot_for_chembl(uniprot_id: str) -> str:
    """
    Normalize UniProt ID for ChEMBL API queries.
    
    GlyGen stores UniProt IDs with isoform suffixes (e.g., 'P26572-1'),
    but ChEMBL stores accessions without the suffix (e.g., 'P26572').
    This function strips the isoform suffix for ChEMBL compatibility.
    
    Args:
        uniprot_id: UniProt accession ID, possibly with isoform suffix.
    
    Returns:
        Normalized UniProt ID without isoform suffix.
    """
    uid = (uniprot_id or "").strip()
    return uid.split("-")[0] if uid else ""


def build_chembl_activity_url(base_url: str, target_chembl_id: str, limit: int = 1000) -> str:
    """
    Build ChEMBL activity API URL.
    
    The URL format follows ChEMBL's expected parameter ordering:
    ?format=json&target_chembl_id=CHEMBLXXXX&limit=1000
    
    Args:
        base_url: ChEMBL API base URL.
        target_chembl_id: ChEMBL target ID.
        limit: Maximum number of results per page.
    
    Returns:
        Properly formatted URL string.
    """
    return f"{base_url}/activity?format=json&target_chembl_id={target_chembl_id}&limit={limit}"


def build_chembl_target_url(base_url: str, uniprot_id: str) -> str:
    """
    Build ChEMBL target search API URL.
    
    Args:
        base_url: ChEMBL API base URL.
        uniprot_id: UniProt accession ID (will be normalized to strip isoform suffix).
    
    Returns:
        Properly formatted URL string.
    """
    normalized_id = normalize_uniprot_for_chembl(uniprot_id)
    return f"{base_url}/target/search?q={normalized_id}&format=json"


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "download.log"))
    ]
)
logger = logging.getLogger(__name__)


def download_glygen_gts():
    """Download GlyGen glycosyltransferase data with retry support."""
    url = config.download.get('glygen', {}).get(
        'gt_url',
        "https://data.glygen.org/ln2data/releases/data/v-2.9.1/reviewed/human_protein_glycosyltransferase.csv"
    )
    output_path = os.path.join(DATA_RAW_DIR, "glygen_glycosyltransferase.csv")
    
    if is_valid_cache(output_path):
        logger.info(f"Using cached file: {output_path}")
        return pd.read_csv(output_path)
    elif os.path.exists(output_path):
        logger.warning(f"{output_path} is too small; deleting and re-downloading.")
        os.remove(output_path)

    logger.info(f"Downloading GlyGen GTs from {url}...")
    
    api_settings = config.get_api_settings('glygen')
    
    try:
        with APIClient(
            max_retries=config.retry.max_retries,
            backoff_factor=config.retry.backoff_seconds,
            timeout=api_settings.timeout,
            rate_limit=api_settings.rate_limit_per_second
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Saved to {output_path}")
            return pd.read_csv(output_path)
            
    except Exception as e:
        logger.error(f"Failed to download GlyGen GTs: {e}")
        return None


def download_glygen_glycan_masterlist():
    """Download GlyGen glycan masterlist with retry support."""
    url = config.download.get('glygen', {}).get(
        'masterlist_url',
        "https://data.glygen.org/ln2data/releases/data/v-2.9.1/reviewed/glycan_masterlist.csv"
    )
    output_path = os.path.join(DATA_RAW_DIR, "glycan_masterlist.csv")
    
    if is_valid_cache(output_path):
        logger.info(f"Using cached file: {output_path}")
        return pd.read_csv(output_path)
    elif os.path.exists(output_path):
        logger.warning(f"{output_path} is too small; deleting and re-downloading.")
        os.remove(output_path)

    logger.info(f"Downloading GlyGen Glycan Masterlist from {url}...")
    
    api_settings = config.get_api_settings('glygen')
    
    try:
        with APIClient(
            max_retries=config.retry.max_retries,
            backoff_factor=config.retry.backoff_seconds,
            timeout=api_settings.timeout,
            rate_limit=api_settings.rate_limit_per_second
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Saved to {output_path}")
            return pd.read_csv(output_path)
            
    except Exception as e:
        logger.error(f"Failed to download Glycan Masterlist: {e}")
        return None


def fetch_glytoucan_structures(glycan_ids):
    """Fetch GlyTouCan structures using SPARQList API with multiprocessing support.
    
    Uses the new GlyCosmos SPARQList gtcid2seqs endpoint:
    GET https://api.glycosmos.org/sparqlist/gtcid2seqs?gtcid={id}
    
    The old /glytoucan/glycan/{id} endpoint is deprecated and no longer returns valid JSON.
    """
    output_path = os.path.join(DATA_RAW_DIR, "glytoucan_structures.tsv")
    
    if is_valid_cache(output_path):
        df = pd.read_csv(output_path, sep="\t")
        if not df.empty:
            logger.info(f"Using cached file: {output_path} ({len(df)} structures)")
            return df
    elif os.path.exists(output_path):
        logger.warning(f"{output_path} is too small; deleting and re-downloading.")
        os.remove(output_path)

    logger.info(f"Fetching GlyTouCan structures for {len(glycan_ids)} glycans...")

    limit = config.get_limit('glytoucan')
    
    if limit is not None and limit < len(glycan_ids):
        target_ids = glycan_ids[:limit]
        logger.info(f"Limited to first {limit} glycans")
    else:
        target_ids = list(glycan_ids)
        logger.info(f"Processing all {len(target_ids)} glycans")

    # Get parallel settings
    parallel_enabled, workers, seed, batch_size = get_parallel_settings()
    
    api_settings = config.get_api_settings('glytoucan')
    
    # Get SPARQList endpoint settings from config
    glytoucan_config = config.download.get('glytoucan', {})
    base_url = glytoucan_config.get('base_url', 'https://api.glycosmos.org/sparqlist')
    gtcid2seqs_endpoint = glytoucan_config.get('gtcid2seqs_endpoint', 'gtcid2seqs')
    min_request_interval_ms = glytoucan_config.get('min_request_interval_ms', 200)
    
    logger.info(f"Using SPARQList endpoint: {base_url}/{gtcid2seqs_endpoint}")
    
    # Build config dict for workers (must be picklable)
    config_dict = {
        'base_url': base_url,
        'gtcid2seqs_endpoint': gtcid2seqs_endpoint,
        'min_request_interval_ms': min_request_interval_ms,
        'max_retries': config.retry.max_retries,
        'backoff_seconds': config.retry.backoff_seconds,
        'timeout': api_settings.timeout,
        'rate_limit': api_settings.rate_limit_per_second,
        'seed': seed,
    }
    
    results = []
    all_failures = []
    parser_stats = {'wurcs_found': 0, 'wurcs_not_found': 0, 'parse_errors': 0}
    
    if parallel_enabled and workers > 1 and len(target_ids) > batch_size:
        # Parallel processing
        logger.info(f"Using parallel processing with {workers} workers")
        
        # Create batches
        batches = list(chunked(target_ids, batch_size))
        tasks = [(batch, config_dict, i) for i, batch in enumerate(batches)]
        
        # Process in parallel
        for batch_result in run_in_process_pool(
            _process_glytoucan_batch,
            tasks,
            len(target_ids),
            "GlyTouCan (parallel)",
            workers
        ):
            if 'error' not in batch_result:
                results.extend(batch_result['results'])
                all_failures.extend(batch_result['failures'])
                # Aggregate parser stats
                stats = batch_result.get('parser_stats', {})
                parser_stats['wurcs_found'] += stats.get('wurcs_found', 0)
                parser_stats['wurcs_not_found'] += stats.get('wurcs_not_found', 0)
                parser_stats['parse_errors'] += stats.get('parse_errors', 0)
    else:
        # Sequential processing (fallback)
        if not parallel_enabled:
            logger.info("Parallel processing disabled, using sequential mode")
        else:
            logger.info(f"Using sequential processing (data size {len(target_ids)} <= batch size {batch_size})")
        
        # Use the same worker function for consistency
        task = (target_ids, config_dict, 0)
        batch_result = _process_glytoucan_batch(task)
        results = batch_result['results']
        all_failures = batch_result['failures']
        parser_stats = batch_result.get('parser_stats', {})

    # Build APIFetchResult from aggregated results
    result = APIFetchResult()
    for r in results:
        result.add_success(r['glycan_id'])
    for f in all_failures:
        result.add_failure(f['id'], f['error_type'], f['message'])

    logger.info(result.summary())
    
    logger.info(f"Parser statistics: {parser_stats}")
    
    if result.failed:
        failed_path = os.path.join(LOG_DIR, "failed_glytoucan_ids.txt")
        result.save_failed_ids(failed_path)

    df = pd.DataFrame(results)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved {len(df)} structures to {output_path}")
    return df


def fetch_uniprot_data(uniprot_ids):
    """Fetch UniProt annotations with proper error handling and multiprocessing support."""
    output_path = os.path.join(DATA_RAW_DIR, "uniprot_annotations.tsv")
    
    if is_valid_cache(output_path):
        logger.info(f"Using cached file: {output_path}")
        return pd.read_csv(output_path, sep="\t")
    elif os.path.exists(output_path):
        logger.warning(f"{output_path} is too small; deleting and re-downloading.")
        os.remove(output_path)

    unique_ids = list(set(uniprot_ids))
    logger.info(f"Fetching UniProt data for {len(unique_ids)} proteins...")

    limit = config.get_limit('uniprot')
    
    if limit is not None and limit < len(unique_ids):
        target_ids = unique_ids[:limit]
        logger.info(f"Limited to first {limit} proteins")
    else:
        target_ids = unique_ids
        logger.info(f"Processing all {len(target_ids)} proteins")

    # Get parallel settings
    parallel_enabled, workers, seed, batch_size = get_parallel_settings()
    
    api_settings = config.get_api_settings('uniprot')
    base_url = config.download.get('uniprot', {}).get(
        'base_url',
        "https://rest.uniprot.org/uniprotkb"
    )
    
    # Build config dict for workers (must be picklable)
    config_dict = {
        'base_url': base_url,
        'max_retries': config.retry.max_retries,
        'backoff_seconds': config.retry.backoff_seconds,
        'timeout': api_settings.timeout,
        'rate_limit': api_settings.rate_limit_per_second,
        'seed': seed,
    }
    
    results = []
    all_failures = []
    
    if parallel_enabled and workers > 1 and len(target_ids) > batch_size:
        # Parallel processing
        logger.info(f"Using parallel processing with {workers} workers")
        
        # Create batches
        batches = list(chunked(target_ids, batch_size))
        tasks = [(batch, config_dict, i) for i, batch in enumerate(batches)]
        
        # Process in parallel
        for batch_result in run_in_process_pool(
            _process_uniprot_batch,
            tasks,
            len(target_ids),
            "UniProt (parallel)",
            workers
        ):
            if 'error' not in batch_result:
                results.extend(batch_result['results'])
                all_failures.extend(batch_result['failures'])
    else:
        # Sequential processing (fallback)
        if not parallel_enabled:
            logger.info("Parallel processing disabled, using sequential mode")
        else:
            logger.info(f"Using sequential processing (data size {len(target_ids)} <= batch size {batch_size})")
        
        # Use the same worker function for consistency
        task = (target_ids, config_dict, 0)
        batch_result = _process_uniprot_batch(task)
        results = batch_result['results']
        all_failures = batch_result['failures']

    # Build APIFetchResult from aggregated results
    result = APIFetchResult()
    for r in results:
        result.add_success(r['uniprot_id'])
    for f in all_failures:
        result.add_failure(f['id'], f['error_type'], f['message'])

    logger.info(result.summary())
    
    if result.failed:
        failed_path = os.path.join(LOG_DIR, "failed_uniprot_ids.txt")
        result.save_failed_ids(failed_path)

    df = pd.DataFrame(results)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved {len(df)} annotations to {output_path}")
    return df


def fetch_glygen_glycan_details(glycan_list):
    """
    Fetch detailed glycan information from GlyGen API with multiprocessing support.
    
    This fetches comprehensive glycan data including:
    - enzyme: enzymes that produce/consume the glycan
    - associated_proteins: proteins associated with the glycan (replaces deprecated usecases API)
    - subsumption: glycan hierarchy relationships (ancestor, descendant, subsumes, subsumed_by)
    - motifs: glycan motif structures
    - residues: glycan residue information
    - classification: glycan classification
    
    The GlyGen detail API (https://api.glygen.org/glycan/detail/{id}) is the ONLY valid
    live API for GlyGen data. The usecases API (/usecases/glycan_to_glycoproteins/) is
    deprecated and always returns empty results.
    
    Args:
        glycan_list: List of GlyTouCan accession IDs.
    
    Returns:
        Dict mapping glycan_id to detail dict, for offline rebuilds and fast debugging.
    """
    output_path = os.path.join(DATA_RAW_DIR, "glygen_glycan_details.json")
    
    if is_valid_cache(output_path):
        logger.info(f"Using cached file: {output_path}")
        with open(output_path, 'r') as f:
            cached_data = json.load(f)
            # Handle both old list format and new dict format
            if isinstance(cached_data, list):
                logger.info("Converting cached list format to dict format")
                return {d['glycan_id']: d for d in cached_data if 'glycan_id' in d}
            return cached_data
    elif os.path.exists(output_path):
        logger.warning(f"{output_path} is too small; deleting and re-downloading.")
        os.remove(output_path)

    logger.info(f"Fetching GlyGen glycan details for {len(glycan_list)} glycans...")

    limit = config.get_limit('glygen')
    seed = config.download.get('glygen', {}).get('random_seed', 42)
    
    if limit is not None and limit < len(glycan_list):
        random.seed(seed)
        sample_list = random.sample(glycan_list, limit)
        logger.info(f"Sampling {limit} glycans (seed={seed})")
    else:
        sample_list = list(glycan_list)
        logger.info(f"Processing all {len(sample_list)} glycans")

    # Get parallel settings
    parallel_enabled, workers, parallel_seed, batch_size = get_parallel_settings()
    
    api_settings = config.get_api_settings('glygen')
    
    # Build config dict for workers (must be picklable)
    config_dict = {
        'max_retries': config.retry.max_retries,
        'backoff_seconds': config.retry.backoff_seconds,
        'timeout': api_settings.timeout,
        'rate_limit': api_settings.rate_limit_per_second,
        'seed': parallel_seed,
    }
    
    all_details = []
    all_failures = []
    
    if parallel_enabled and workers > 1 and len(sample_list) > batch_size:
        # Parallel processing
        logger.info(f"Using parallel processing with {workers} workers")
        
        # Create batches
        batches = list(chunked(sample_list, batch_size))
        tasks = [(batch, config_dict, i) for i, batch in enumerate(batches)]
        
        # Process in parallel
        for batch_result in run_in_process_pool(
            _process_glygen_glycan_detail_batch,
            tasks,
            len(sample_list),
            "Glycan Details (parallel)",
            workers
        ):
            if 'error' not in batch_result:
                all_details.extend(batch_result['glycan_details'])
                all_failures.extend(batch_result['failures'])
    else:
        # Sequential processing (fallback)
        if not parallel_enabled:
            logger.info("Parallel processing disabled, using sequential mode")
        else:
            logger.info(f"Using sequential processing (data size {len(sample_list)} <= batch size {batch_size})")
        
        # Use the same worker function for consistency
        task = (sample_list, config_dict, 0)
        batch_result = _process_glygen_glycan_detail_batch(task)
        all_details = batch_result['glycan_details']
        all_failures = batch_result['failures']

    # Build APIFetchResult from aggregated results
    result = APIFetchResult()
    for detail in all_details:
        result.add_success(detail['glycan_id'])
    for f in all_failures:
        result.add_failure(f['id'], f['error_type'], f['message'])

    logger.info(result.summary())
    
    # Diagnostics for associated_proteins extraction quality
    details_with_assoc = 0
    assoc_total = 0
    assoc_source_counts = {}
    for detail in all_details:
        assoc = detail.get('associated_proteins', [])
        if isinstance(assoc, list) and len(assoc) > 0:
            details_with_assoc += 1
            assoc_total += len(assoc)
        src_key = detail.get('associated_proteins_source_key', 'none')
        assoc_source_counts[src_key] = assoc_source_counts.get(src_key, 0) + 1

    logger.info(
        "Associated proteins summary: details_with_assoc=%d/%d, total_records=%d, source_keys=%s",
        details_with_assoc,
        len(all_details),
        assoc_total,
        assoc_source_counts,
    )
    if len(all_details) > 0 and details_with_assoc == 0:
        logger.warning(
            "No associated_proteins were extracted from GlyGen detail responses. "
            "This may indicate upstream API responses contain no protein links "
            "or response field schema drift."
        )
    
    if result.failed:
        failed_path = os.path.join(LOG_DIR, "failed_glycan_detail_ids.txt")
        result.save_failed_ids(failed_path)

    # Convert list to dict mapping for easier lookup and offline rebuilds
    # Format: {"G01818TY": {full detail dict}, "G01820WV": {full detail dict}, ...}
    details_dict = {d['glycan_id']: d for d in all_details}
    
    # Save as JSON to preserve nested structure
    with open(output_path, 'w') as f:
        json.dump(details_dict, f, indent=2)
    logger.info(f"Saved {len(details_dict)} glycan details to {output_path}")
    
    return details_dict


def fetch_chembl_data(enzymes_map):
    """Fetch ChEMBL inhibitor data with proper error handling."""
    output_path = os.path.join(DATA_RAW_DIR, "chembl_gt_inhibitors.tsv")
    
    if is_valid_cache(output_path):
        logger.info(f"Using cached file: {output_path}")
        return pd.read_csv(output_path, sep="\t")
    elif os.path.exists(output_path):
        logger.warning(f"{output_path} is too small; deleting and re-downloading.")
        os.remove(output_path)

    logger.info(f"Fetching ChEMBL data for {len(enzymes_map)} enzymes...")

    limit = config.get_limit('chembl')
    enzyme_items = list(enzymes_map.items())
    
    if limit is not None and limit < len(enzyme_items):
        enzyme_items = enzyme_items[:limit]
        logger.info(f"Limited to first {limit} enzymes")
    else:
        logger.info(f"Processing all {len(enzyme_items)} enzymes")

    all_activities = []
    result = APIFetchResult()
    
    api_settings = config.get_api_settings('chembl')
    base_url = config.download.get('chembl', {}).get(
        'base_url',
        "https://www.ebi.ac.uk/chembl/api/data"
    )
    
    targets_found = 0
    activities_total = 0
    
    with APIClient(
        max_retries=config.retry.max_retries,
        backoff_factor=config.retry.backoff_seconds,
        timeout=api_settings.timeout,
        rate_limit=api_settings.rate_limit_per_second
    ) as client:
        for uid, gene in tqdm(enzyme_items, desc="ChEMBL"):
            base_uid = normalize_uniprot_for_chembl(uid)
            if not base_uid:
                result.add_failure(uid, 'invalid_id', 'Empty or invalid UniProt ID')
                continue
            
            target_url = build_chembl_target_url(base_url, uid)
            logger.debug(f"Searching ChEMBL for {base_uid} (original: {uid})")
            
            try:
                target_data = client.get_json(target_url, validate_content_type=True)
                
                if target_data is None:
                    result.add_failure(uid, 'no_target_response', 'No response for target search')
                    continue
                
                if not isinstance(target_data, dict):
                    result.add_failure(uid, 'invalid_response', 'ChEMBL returned non-dict response')
                    continue
                
                if 'targets' not in target_data:
                    result.add_failure(uid, 'invalid_response', 'ChEMBL response missing "targets" key')
                    continue

                targets = target_data.get('targets', [])
                target_chembl_ids = []

                for t in targets:
                    components = t.get('target_components', [])
                    for c in components:
                        if (c.get('component_type') == 'PROTEIN' and 
                            c.get('accession') == base_uid):
                            tid = t.get('target_chembl_id')
                            if tid:
                                target_chembl_ids.append(tid)
                            break

                if not target_chembl_ids:
                    result.add_failure(uid, 'no_target_found', f'No matching ChEMBL target found for {base_uid}')
                    continue
                
                targets_found += 1
                logger.debug(f"Found {len(target_chembl_ids)} ChEMBL target(s) for {base_uid}: {target_chembl_ids}")

                activities_found = 0
                for target_chembl_id in target_chembl_ids:
                    act_url = build_chembl_activity_url(base_url, target_chembl_id, limit=1000)
                    
                    while act_url:
                        act_data = client.get_json(act_url, validate_content_type=True)
                        
                        if act_data is None:
                            logger.warning(f"No activity data for {uid} ({target_chembl_id})")
                            break
                        
                        if not isinstance(act_data, dict) or 'activities' not in act_data:
                            logger.warning(f"Invalid activity response for {uid} ({target_chembl_id})")
                            break
                            
                        activities = act_data.get('activities', [])

                        for act in activities:
                            if act.get('standard_type') in ['IC50', 'Ki', 'Kd', 'Inhibition']:
                                all_activities.append({
                                    'enzyme_uniprot_id': uid,
                                    'target_chembl_id': target_chembl_id,
                                    'compound_chembl_id': act.get('molecule_chembl_id'),
                                    'compound_name': act.get('molecule_pref_name'),
                                    'type': act.get('standard_type'),
                                    'value': act.get('standard_value'),
                                    'units': act.get('standard_units')
                                })
                                activities_found += 1

                        meta = act_data.get('page_meta', {})
                        next_page = meta.get('next')
                        act_url = f"https://www.ebi.ac.uk{next_page}" if next_page else None

                if activities_found > 0:
                    result.add_success(uid)
                    activities_total += activities_found
                    logger.debug(f"Found {activities_found} activities for {uid}")
                else:
                    result.add_failure(uid, 'no_activities', 'No inhibitor activities found')

            except ValueError as e:
                logger.error(f"JSON validation error for {uid}: {e}")
                result.add_failure(uid, 'json_error', str(e))
            except Exception as e:
                logger.error(f"Error fetching ChEMBL data for {uid}: {e}")
                result.add_failure(uid, 'exception', str(e))
    
    logger.info(f"ChEMBL fetch complete: {targets_found} targets found, {activities_total} total activities")

    logger.info(result.summary())
    
    if result.failed:
        failed_path = os.path.join(LOG_DIR, "failed_chembl_ids.txt")
        result.save_failed_ids(failed_path)

    df = pd.DataFrame(all_activities)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved {len(df)} inhibitor activities to {output_path}")
    return df


def test_chembl_for_uniprot(uniprot_id: str):
    """
    Test ChEMBL fetch for a single UniProt ID.
    
    This is a standalone test function to validate ChEMBL API connectivity
    and response parsing for a given UniProt accession.
    
    Args:
        uniprot_id: UniProt accession ID to test.
    """
    from utils.api_client import APIClient
    
    base_url = "https://www.ebi.ac.uk/chembl/api/data"
    base_uid = normalize_uniprot_for_chembl(uniprot_id)
    
    print(f"\n=== ChEMBL Test for {uniprot_id} ===")
    print(f"Normalized ID: {base_uid}")
    
    with APIClient(max_retries=3, backoff_factor=2, timeout=30, rate_limit=10) as client:
        target_url = build_chembl_target_url(base_url, uniprot_id)
        print(f"\nTarget search URL: {target_url}")
        
        target_data = client.get_json(target_url, validate_content_type=True)
        
        if target_data is None:
            print("ERROR: No response from ChEMBL target search")
            return
        
        if 'targets' not in target_data:
            print(f"ERROR: Response missing 'targets' key. Keys: {list(target_data.keys())}")
            return
        
        targets = target_data.get('targets', [])
        print(f"Total targets returned: {len(targets)}")
        
        target_chembl_ids = []
        for t in targets:
            components = t.get('target_components', [])
            for c in components:
                if c.get('component_type') == 'PROTEIN' and c.get('accession') == base_uid:
                    tid = t.get('target_chembl_id')
                    if tid:
                        target_chembl_ids.append(tid)
                        print(f"  MATCH: {tid} - {t.get('pref_name')}")
        
        if not target_chembl_ids:
            print(f"No matching ChEMBL targets found for accession {base_uid}")
            return
        
        print(f"\nMatching target IDs: {target_chembl_ids}")
        
        total_activities = 0
        for target_chembl_id in target_chembl_ids:
            act_url = build_chembl_activity_url(base_url, target_chembl_id, limit=10)
            print(f"\nActivity URL: {act_url}")
            
            act_data = client.get_json(act_url, validate_content_type=True)
            
            if act_data is None or 'activities' not in act_data:
                print(f"  No activities for {target_chembl_id}")
                continue
            
            page_meta = act_data.get('page_meta', {})
            total_count = page_meta.get('total_count', 0)
            activities = act_data.get('activities', [])
            
            print(f"  Total activities: {total_count}")
            total_activities += total_count
            
            print(f"  Sample activities (first 3):")
            for act in activities[:3]:
                print(f"    {act.get('molecule_chembl_id')}: {act.get('standard_type')}={act.get('standard_value')} {act.get('standard_units')}")
        
        print(f"\n=== Summary ===")
        print(f"UniProt ID: {uniprot_id} (normalized: {base_uid})")
        print(f"ChEMBL targets: {len(target_chembl_ids)}")
        print(f"Total activities: {total_activities}")


def download_uniprot_sites(protein_list, output_path=None):
    """
    Fetch UniProt glycosylation site data for all proteins in protein_list.
    Uses scripts/utils/uniprot_sites.py.
    Saves raw JSON records to output_path.
    
    Note: Evidence filtering is handled in clean_uniprot_sites(), not here.
    The download stage fetches all CARBOHYD features with valid evidence codes.
    
    Args:
        protein_list: List of UniProt accession IDs to fetch sites for.
        output_path: Path to save raw JSON records. Defaults to data_raw/uniprot_sites_raw.json.
    
    Returns:
        List of raw site records or None if failed.
    """
    if output_path is None:
        output_path = os.path.join(DATA_RAW_DIR, "uniprot_sites_raw.json")
    
    if not config.site_data.enable_uniprot_sites:
        logger.info("UniProt sites download disabled in config")
        return None
    
    if is_valid_cache(output_path, min_bytes=100):
        logger.info(f"Using cached UniProt sites file: {output_path}")
        with open(output_path, 'r') as f:
            return json.load(f)
    elif os.path.exists(output_path):
        logger.warning(f"{output_path} is too small; deleting and re-downloading.")
        os.remove(output_path)
    
    if not protein_list:
        logger.warning("No proteins provided for UniProt site extraction")
        return []
    
    logger.info(f"Fetching UniProt glycosylation sites for {len(protein_list)} proteins...")
    
    from utils.uniprot_sites import UniProtSiteExtractor
    
    api_settings = config.get_api_settings('uniprot')
    extractor = UniProtSiteExtractor(
        max_retries=config.retry.max_retries,
        timeout=api_settings.timeout,
        rate_limit=api_settings.rate_limit_per_second
    )
    
    limit = config.get_limit('uniprot')
    if limit:
        protein_list = protein_list[:limit]
    
    sites = extractor.fetch_and_extract(protein_list)
    
    all_sites = []
    for site in sites:
        all_sites.append({
            'uniprot_id': site.uniprot_id,
            'site_position': site.site_position,
            'site_residue': site.site_residue,
            'site_type': site.site_type,
            'evidence_code': site.evidence_code,
            'evidence_type': site.evidence_type,
            'description': site.description,
            'source': 'UniProt'
        })
    
    logger.info(f"Extracted {len(all_sites)} sites from {len(protein_list)} proteins")
    
    with open(output_path, 'w') as f:
        json.dump(all_sites, f, indent=2)
    logger.info(f"Saved {len(all_sites)} raw site records to {output_path}")
    
    stats = extractor.get_statistics()
    logger.info(f"UniProt site extraction stats: {stats.summary()}")
    
    return all_sites


def download_ptmcode_data(output_dir=None):
    """
    Download PTMCode v3 data (sites + crosstalk edges).
    Uses scripts/utils/ptmcode_loader.py.
    
    This function will:
    1. Check for PTMCode2 raw association files and convert them if present
    2. Fall back to creating placeholder files if no raw data is available
    
    PTMCode2 raw files expected:
    - PTMcode2_associations_within_proteins.txt
    - PTMcode2_associations_between_proteins.txt
    
    Args:
        output_dir: Directory to save PTMCode raw files. Defaults to data_raw/ptmcode_raw/.
    
    Returns:
        Dict with paths to downloaded files or None if disabled.
    """
    if output_dir is None:
        output_dir = os.path.join(DATA_RAW_DIR, "ptmcode_raw")
    
    if not config.site_data.enable_ptmcode:
        logger.info("PTMCode download disabled in config")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    sites_path = os.path.join(output_dir, "sites_raw.tsv")
    edges_path = os.path.join(output_dir, "edges_raw.tsv")
    metadata_path = os.path.join(output_dir, "metadata.json")
    
    ptmcode2_within = os.path.join(output_dir, "PTMcode2_associations_within_proteins.txt")
    ptmcode2_between = os.path.join(output_dir, "PTMcode2_associations_between_proteins.txt")
    
    ptmcode2_files_exist = os.path.exists(ptmcode2_within) or os.path.exists(ptmcode2_between)
    
    if ptmcode2_files_exist:
        logger.info("Detected PTMCode2 raw association files. Converting to standardized format...")
        
        from utils.ptmcode_loader import convert_ptmcode2_to_raw
        
        stats = convert_ptmcode2_to_raw(
            input_within=ptmcode2_within,
            input_between=ptmcode2_between,
            output_sites=sites_path,
            output_edges=edges_path,
            min_score=config.site_data.ptmcode_min_score,
        )
        
        metadata = {
            'source': 'PTMCode2',
            'url': 'https://ptmcode.embl.de/',
            'download_date': datetime.now().isoformat(),
            'min_score': config.site_data.ptmcode_min_score,
            'conversion_stats': {
                'lines_processed': stats.lines_processed,
                'sites_extracted': stats.sites_extracted,
                'sites_deduplicated': stats.sites_deduplicated,
                'edges_within': stats.edges_within,
                'edges_between': stats.edges_between,
            },
            'note': 'Converted from PTMCode2 raw association files.'
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"PTMCode2 conversion complete: {stats.sites_deduplicated} sites, {stats.edges_within + stats.edges_between} edges")
        
        return {
            'sites_path': sites_path,
            'edges_path': edges_path,
            'metadata_path': metadata_path
        }
    
    if is_valid_cache(sites_path, min_bytes=100) and is_valid_cache(edges_path, min_bytes=100):
        logger.info(f"Using cached PTMCode files in {output_dir}")
        return {
            'sites_path': sites_path,
            'edges_path': edges_path,
            'metadata_path': metadata_path
        }
    
    logger.info("PTMCode data download...")
    logger.info("Note: PTMCode v3 data should be downloaded from https://ptmcode.embl.de/")
    logger.info("To enable automatic conversion, place PTMCode2 files in data_raw/ptmcode_raw/:")
    logger.info("  - PTMcode2_associations_within_proteins.txt")
    logger.info("  - PTMcode2_associations_between_proteins.txt")
    logger.info("Creating placeholder files for manual population.")
    
    if not os.path.exists(sites_path):
        with open(sites_path, 'w') as f:
            f.write("uniprot_id\tsite_position\tsite_residue\tptm_type\tsource\n")
        logger.info(f"Created placeholder sites file: {sites_path}")
    
    if not os.path.exists(edges_path):
        with open(edges_path, 'w') as f:
            f.write("uniprot_id_a\tsite_position_a\tresidue_a\tptm_type_a\t")
            f.write("uniprot_id_b\tsite_position_b\tresidue_b\tptm_type_b\t")
            f.write("score\tedge_type\tsource\n")
        logger.info(f"Created placeholder edges file: {edges_path}")
    
    metadata = {
        'source': 'PTMCode v3',
        'url': 'https://ptmcode.embl.de/',
        'download_date': datetime.now().isoformat(),
        'min_score': config.site_data.ptmcode_min_score,
        'note': 'Placeholder files created. Place PTMCode2 raw files in this directory and re-run.'
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"PTMCode placeholder files created in {output_dir}")
    
    return {
        'sites_path': sites_path,
        'edges_path': edges_path,
        'metadata_path': metadata_path
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download data from GlyGen, GlyTouCan, UniProt, ChEMBL, and site-level PTM APIs."
    )
    parser.add_argument(
        "--stage",
        choices=["all", "glygen", "glytoucan", "uniprot", "chembl", "sites", "ptmcode"],
        default="all",
        help="Limit download to a specific data source. 'sites' fetches UniProt glycosylation sites. 'ptmcode' fetches PTMCode crosstalk data."
    )
    parser.add_argument(
        "--test-chembl",
        metavar="UNIPROT_ID",
        help="Test ChEMBL fetch for a single UniProt ID and exit. Example: --test-chembl P26572"
    )
    return parser.parse_args()


def main():
    """Main entry point for data download pipeline."""
    args = parse_args()
    
    if args.test_chembl:
        test_chembl_for_uniprot(args.test_chembl)
        return
    
    stage = args.stage
    
    start_time = datetime.now()
    logger.info(f"Starting data download at {start_time} (stage={stage})")
    logger.info(f"Configuration: limits={config.api_limits}, retry={config.retry}")
    
    # Always download GlyGen GTs first (needed for chembl stage and others)
    gt_df = download_glygen_gts()
    
    if gt_df is None:
        logger.error("Failed to download GlyGen GTs. Aborting.")
        return

    col_name = 'uniprotkb_canonical_ac'
    if col_name not in gt_df.columns:
        logger.error(f"Missing {col_name} in GT data. Aborting.")
        return
        
    gt_uniprot_ids = gt_df[col_name].dropna().unique().tolist()
    logger.info(f"Found {len(gt_uniprot_ids)} unique UniProt IDs from GlyGen GTs")

    gt_map = {}
    if 'gene_symbol' in gt_df.columns:
        gt_map = pd.Series(gt_df.gene_symbol.values, index=gt_df[col_name]).to_dict()

    # Download glycan masterlist (needed for glygen, glytoucan stages)
    glycan_list = []
    if stage in ("all", "glygen", "glytoucan"):
        masterlist_df = download_glygen_glycan_masterlist()
        if masterlist_df is not None and 'glytoucan_ac' in masterlist_df.columns:
            glycan_list = masterlist_df['glytoucan_ac'].dropna().unique().tolist()
            logger.info(f"Found {len(glycan_list)} unique glycan IDs from masterlist")

    # GlyGen glycan details (replaces deprecated usecases API)
    # The detail API is the ONLY valid live API for GlyGen data
    # It provides associated_proteins which replaces glygen_glycan_protein.tsv
    gp_uniprot_ids = []
    if stage in ("all", "glygen"):
        logger.info("Fetching GlyGen glycan details (includes associated proteins)...")
        glycan_details = fetch_glygen_glycan_details(glycan_list)
        
        # Extract UniProt IDs from associated_proteins for UniProt stage
        if glycan_details:
            for glycan_id, detail in glycan_details.items():
                for protein in detail.get('associated_proteins', []):
                    uniprot_id = protein.get('uniprot_canonical_ac')
                    if uniprot_id:
                        gp_uniprot_ids.append(uniprot_id)
            gp_uniprot_ids = list(set(gp_uniprot_ids))
            logger.info(f"Found {len(gp_uniprot_ids)} unique UniProt IDs from glycan-protein associations")

    # GlyTouCan structures
    if stage in ("all", "glytoucan"):
        fetch_glytoucan_structures(glycan_list)

    # UniProt annotations
    if stage in ("all", "uniprot"):
        all_uniprot_ids = list(set(gt_uniprot_ids + gp_uniprot_ids))
        logger.info(f"Total unique UniProt IDs to fetch: {len(all_uniprot_ids)}")
        fetch_uniprot_data(all_uniprot_ids)

    # ChEMBL inhibitor data
    if stage in ("all", "chembl"):
        fetch_chembl_data(gt_map)

    # UniProt glycosylation sites
    if stage in ("all", "sites"):
        if config.site_data.enable_uniprot_sites:
            all_uniprot_ids = list(set(gt_uniprot_ids + gp_uniprot_ids))
            if all_uniprot_ids:
                logger.info(f"Fetching UniProt glycosylation sites for {len(all_uniprot_ids)} proteins...")
                download_uniprot_sites(all_uniprot_ids)
            else:
                logger.warning("No UniProt IDs available for site extraction")
        else:
            logger.info("UniProt sites download disabled in config")

    # PTMCode crosstalk data
    if stage in ("all", "ptmcode"):
        if config.site_data.enable_ptmcode:
            logger.info("Downloading PTMCode crosstalk data...")
            download_ptmcode_data()
        else:
            logger.info("PTMCode download disabled in config")

    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Data download completed at {end_time} (duration: {duration})")


if __name__ == "__main__":
    main()
