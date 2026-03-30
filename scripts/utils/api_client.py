#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
api_client.py

HTTP client with retry functionality and rate limiting for glycoMusubi pipeline.
Provides robust API access with exponential backoff and detailed error logging.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import defaultdict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class APIFetchResult:
    """Container for API fetch results with success/failure tracking."""
    success: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)
    errors: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    data: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_success(self, item_id: str, data: Optional[Dict] = None):
        """Record a successful fetch."""
        self.success.append(item_id)
        if data:
            self.data.append(data)
    
    def add_failure(self, item_id: str, error_type: str, error_msg: str = ""):
        """Record a failed fetch."""
        self.failed.append(item_id)
        self.errors[error_type].append(f"{item_id}: {error_msg}" if error_msg else item_id)
    
    def summary(self) -> str:
        """Generate a summary of fetch results."""
        lines = [
            f"Fetch Results: {len(self.success)} success, {len(self.failed)} failed",
        ]
        if self.errors:
            lines.append("Errors by type:")
            for error_type, items in self.errors.items():
                lines.append(f"  {error_type}: {len(items)} items")
        return "\n".join(lines)
    
    def save_failed_ids(self, filepath: str):
        """Save failed IDs to a file for later retry."""
        if self.failed:
            with open(filepath, 'w') as f:
                f.write("\n".join(self.failed))
            logger.info(f"Saved {len(self.failed)} failed IDs to {filepath}")


def create_session_with_retry(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: Optional[List[int]] = None,
    timeout: int = 30
) -> requests.Session:
    """
    Create a requests Session with automatic retry functionality.
    
    Args:
        max_retries: Maximum number of retry attempts.
        backoff_factor: Factor for exponential backoff (wait = backoff_factor * 2^attempt).
        status_forcelist: HTTP status codes that trigger a retry.
        timeout: Default timeout for requests.
    
    Returns:
        Configured requests.Session object.
    """
    if status_forcelist is None:
        status_forcelist = [429, 500, 502, 503, 504]
    
    session = requests.Session()
    
    retry_strategy = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET", "POST", "HEAD", "OPTIONS"],
        raise_on_status=False,
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    session.timeout = timeout
    
    return session


def fetch_with_retry(
    session: requests.Session,
    url: str,
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    timeout: int = 30,
    headers: Optional[Dict[str, str]] = None,
    method: str = "GET",
    **kwargs
) -> requests.Response:
    """
    Fetch a URL with manual retry logic and exponential backoff.
    
    Args:
        session: requests.Session to use.
        url: URL to fetch.
        max_retries: Maximum number of retry attempts.
        backoff_factor: Factor for exponential backoff.
        timeout: Request timeout in seconds.
        headers: Optional headers to include.
        method: HTTP method (GET, POST, etc.).
        **kwargs: Additional arguments passed to session.request().
    
    Returns:
        requests.Response object.
    
    Raises:
        requests.exceptions.RetryError: If all retries are exhausted.
        requests.exceptions.RequestException: For other request errors.
    """
    if headers is None:
        headers = {}
    
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            response = session.request(
                method=method,
                url=url,
                headers=headers,
                timeout=timeout,
                **kwargs
            )
            
            if response.status_code == 429:
                wait_time = backoff_factor ** attempt
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    try:
                        wait_time = max(wait_time, int(retry_after))
                    except ValueError:
                        pass
                logger.warning(f"Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
                continue
            
            if response.status_code >= 500:
                wait_time = backoff_factor ** attempt
                logger.warning(f"Server error {response.status_code}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            return response
            
        except requests.exceptions.Timeout as e:
            last_exception = e
            wait_time = backoff_factor ** attempt
            logger.warning(f"Timeout for {url}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
            time.sleep(wait_time)
            
        except requests.exceptions.ConnectionError as e:
            last_exception = e
            wait_time = backoff_factor ** attempt
            logger.warning(f"Connection error for {url}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
            time.sleep(wait_time)
            
        except requests.exceptions.HTTPError as e:
            last_exception = e
            if e.response is not None and e.response.status_code < 500:
                raise
            wait_time = backoff_factor ** attempt
            logger.warning(f"HTTP error for {url}: {e}. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
            time.sleep(wait_time)
    
    error_msg = f"Failed after {max_retries} retries: {url}"
    if last_exception:
        error_msg += f" (last error: {last_exception})"
    
    raise requests.exceptions.RetryError(error_msg)


class RateLimiter:
    """Simple rate limiter to control API request frequency."""
    
    def __init__(self, requests_per_second: float = 10.0):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum requests per second.
        """
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()


class APIClient:
    """
    High-level API client with retry, rate limiting, and error tracking.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        timeout: int = 30,
        rate_limit: float = 10.0
    ):
        """
        Initialize API client.
        
        Args:
            max_retries: Maximum retry attempts.
            backoff_factor: Exponential backoff factor.
            timeout: Request timeout in seconds.
            rate_limit: Maximum requests per second.
        """
        self.session = create_session_with_retry(
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            timeout=timeout
        )
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.rate_limiter = RateLimiter(rate_limit)
    
    def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> requests.Response:
        """
        Make a GET request with rate limiting and retry.
        
        Args:
            url: URL to fetch.
            headers: Optional headers.
            **kwargs: Additional arguments.
        
        Returns:
            requests.Response object.
        """
        self.rate_limiter.wait()
        return fetch_with_retry(
            session=self.session,
            url=url,
            max_retries=self.max_retries,
            backoff_factor=self.backoff_factor,
            timeout=self.timeout,
            headers=headers,
            method="GET",
            **kwargs
        )
    
    def get_json(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        validate_content_type: bool = True,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Make a GET request and parse JSON response.
        
        Args:
            url: URL to fetch.
            headers: Optional headers.
            validate_content_type: If True, validate that response Content-Type contains 'json'.
            **kwargs: Additional arguments.
        
        Returns:
            Parsed JSON data or None if parsing fails.
        
        Raises:
            ValueError: If validate_content_type is True and response is not JSON.
        """
        try:
            response = self.get(url, headers=headers, **kwargs)
            
            # Validate Content-Type header
            if validate_content_type:
                content_type = response.headers.get("Content-Type", "")
                if "json" not in content_type.lower():
                    logger.error(f"Non-JSON response from {url}: Content-Type={content_type}")
                    raise ValueError(f"Non-JSON response from {url}: Content-Type={content_type}")
            
            return response.json()
        except requests.exceptions.JSONDecodeError as e:
            logger.error(f"JSON decode error for {url}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {url}: {e}")
            return None
    
    def close(self):
        """Close the session."""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
