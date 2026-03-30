#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
config_loader.py

Configuration file loader for glycoMusubi pipeline.
Reads settings from config.yaml and provides typed access.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetryConfig:
    max_retries: int = 3
    backoff_seconds: int = 2


@dataclass
class APILimits:
    glygen_max: Optional[int] = None
    uniprot_max: Optional[int] = None
    chembl_max: Optional[int] = None
    glytoucan_max: Optional[int] = None


@dataclass
class APISettings:
    rate_limit_per_second: int = 10
    timeout: int = 30


@dataclass
class DirectoryConfig:
    raw_data: str = "data_raw"
    clean_data: str = "data_clean"
    kg_output: str = "kg"
    logs: str = "logs"


@dataclass
class ParallelConfig:
    enabled: bool = True
    workers: Optional[int] = None  # None = auto-detect CPU count
    seed: int = 42
    batch_size: int = 100


@dataclass
class SiteDataConfig:
    enable_uniprot_sites: bool = True
    enable_ptmcode: bool = True
    uniprot_evidence_filter: List[str] = field(default_factory=lambda: [
        "ECO:0000269",  # experimental
        "ECO:0000303",  # expert curated
        "ECO:0000250",  # manual assertion
    ])
    ptmcode_min_score: float = 0.30


@dataclass
class Config:
    api_limits: APILimits = field(default_factory=APILimits)
    retry: RetryConfig = field(default_factory=RetryConfig)
    directories: DirectoryConfig = field(default_factory=DirectoryConfig)
    api_settings: Dict[str, APISettings] = field(default_factory=dict)
    download: Dict[str, Any] = field(default_factory=dict)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    site_data: SiteDataConfig = field(default_factory=SiteDataConfig)
    
    def get_limit(self, api_name: str) -> Optional[int]:
        """Get the limit for a specific API."""
        limit_map = {
            'glygen': self.api_limits.glygen_max,
            'uniprot': self.api_limits.uniprot_max,
            'chembl': self.api_limits.chembl_max,
            'glytoucan': self.api_limits.glytoucan_max,
        }
        return limit_map.get(api_name.lower())
    
    def get_api_settings(self, api_name: str) -> APISettings:
        """Get API settings for a specific API."""
        return self.api_settings.get(api_name.lower(), APISettings())


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml. If None, searches in default locations.
    
    Returns:
        Config object with all settings.
    """
    if config_path is None:
        search_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"),
            os.path.join(os.getcwd(), "config.yaml"),
            "config.yaml",
        ]
        for path in search_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path is None or not os.path.exists(config_path):
        logger.warning("config.yaml not found, using default settings")
        return Config()
    
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    
    if raw is None:
        return Config()
    
    api_limits = APILimits(
        glygen_max=raw.get('api', {}).get('glygen_max'),
        uniprot_max=raw.get('api', {}).get('uniprot_max'),
        chembl_max=raw.get('api', {}).get('chembl_max'),
        glytoucan_max=raw.get('api', {}).get('glytoucan_max'),
    )
    
    retry_config = RetryConfig(
        max_retries=raw.get('retry', {}).get('max_retries', 3),
        backoff_seconds=raw.get('retry', {}).get('backoff_seconds', 2),
    )
    
    directories = DirectoryConfig(
        raw_data=raw.get('directories', {}).get('raw_data', 'data_raw'),
        clean_data=raw.get('directories', {}).get('clean_data', 'data_clean'),
        kg_output=raw.get('directories', {}).get('kg_output', 'kg'),
        logs=raw.get('directories', {}).get('logs', 'logs'),
    )
    
    api_settings = {}
    for api_name, settings in raw.get('api_settings', {}).items():
        api_settings[api_name.lower()] = APISettings(
            rate_limit_per_second=settings.get('rate_limit_per_second', 10),
            timeout=settings.get('timeout', 30),
        )
    
    download = raw.get('download', {})
    
    parallel_raw = raw.get('parallel', {})
    parallel_config = ParallelConfig(
        enabled=parallel_raw.get('enabled', True),
        workers=parallel_raw.get('workers'),
        seed=parallel_raw.get('seed', 42),
        batch_size=parallel_raw.get('batch_size', 100),
    )
    
    site_data_raw = raw.get('site_data', {})
    site_data_config = SiteDataConfig(
        enable_uniprot_sites=site_data_raw.get('enable_uniprot_sites', True),
        enable_ptmcode=site_data_raw.get('enable_ptmcode', True),
        uniprot_evidence_filter=site_data_raw.get('uniprot_evidence_filter', [
            "ECO:0000269", "ECO:0000303", "ECO:0000250"
        ]),
        ptmcode_min_score=site_data_raw.get('ptmcode_min_score', 0.30),
    )
    
    return Config(
        api_limits=api_limits,
        retry=retry_config,
        directories=directories,
        api_settings=api_settings,
        download=download,
        parallel=parallel_config,
        site_data=site_data_config,
    )


_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance (singleton pattern)."""
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance


def reload_config(config_path: Optional[str] = None) -> Config:
    """Reload configuration from file."""
    global _config_instance
    _config_instance = load_config(config_path)
    return _config_instance
