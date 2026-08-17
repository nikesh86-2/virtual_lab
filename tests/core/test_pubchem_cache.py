"""
Tests for Priority 6: PubChem retries.

Verifies:
- _pubchem_get_json retries on transient failures
- _pubchem_get_json raises on permanent failures after max retries
- _pubchem_get_json succeeds on first call for valid CID
- retry count is configurable via environment variable
"""

import json
import time
from http.client import HTTPException
from unittest.mock import MagicMock, patch

import pytest
import requests


class TestPubChemRetry:
    """Test PubChem retry logic."""

    def test_retries_on_transient_failure(self):
        """Should retry on transient HTTP errors."""
        from core.small_molecule_prep import _pubchem_get_json

        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise requests.exceptions.ConnectionError("Transient error")
            # Return a valid response on 3rd attempt
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"CID": 12345, "MolecularFormula": "C6H12O6"}
            return mock_response

        with patch("requests.get", side_effect=mock_get):
            result = _pubchem_get_json("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/12345/json")

        assert call_count == 3
        assert result["CID"] == 12345

    def test_raises_after_max_retries(self):
        """Should return None after exhausting retries on persistent failures (Priority 1 fix)."""
        from core.small_molecule_prep import _pubchem_get_json

        with patch(
            "requests.get",
            side_effect=requests.exceptions.ConnectionError("Persistent error"),
        ):
            # Priority 1 fix: PubChem failures now return None instead of raising
            result = _pubchem_get_json("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/99999/json")
            assert result is None

    def test_succeeds_on_first_call(self):
        """Should succeed immediately for valid CID."""
        from core.small_molecule_prep import _pubchem_get_json

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"CID": 2244, "MolecularFormula": "C8H10N4O2"}

        with patch("requests.get", return_value=mock_response) as mock_get:
            result = _pubchem_get_json("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/json")

        assert mock_get.call_count == 1
        assert result["CID"] == 2244

    def test_retries_on_429_rate_limit(self):
        """Should retry on 429 rate limit responses."""
        from core.small_molecule_prep import _pubchem_get_json

        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                mock_response = MagicMock()
                mock_response.status_code = 429
                mock_response.headers = {"Retry-After": "1"}
                raise requests.exceptions.HTTPError(response=mock_response)
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"CID": 5962}
            return mock_response

        with patch("requests.get", side_effect=mock_get):
            result = _pubchem_get_json("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/5962/json")

        assert call_count == 2
        assert result["CID"] == 5962

    def test_retries_on_500_server_error(self):
        """Should retry on 500 Internal Server Error."""
        from core.small_molecule_prep import _pubchem_get_json

        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                mock_response = MagicMock()
                mock_response.status_code = 500
                raise requests.exceptions.HTTPError(response=mock_response)
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"CID": 2244}
            return mock_response

        with patch("requests.get", side_effect=mock_get):
            result = _pubchem_get_json("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/json")

        assert call_count == 2

    def test_timeout_is_respected(self):
        """Should use timeout on requests."""
        from core.small_molecule_prep import _pubchem_get_json

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"CID": 2244}

        with patch("requests.get", return_value=mock_response) as mock_get:
            _pubchem_get_json("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/json")

        call_kwargs = mock_get.call_args[1]
        assert "timeout" in call_kwargs
        assert call_kwargs["timeout"] > 0

    def test_user_agent_is_sent(self):
        """Should send User-Agent header."""
        from core.small_molecule_prep import _pubchem_get_json

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"CID": 2244}

        with patch("requests.get", return_value=mock_response) as mock_get:
            _pubchem_get_json("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/json")

        call_kwargs = mock_get.call_args[1]
        headers = call_kwargs.get("headers", {})
        assert "User-Agent" in headers or "user-agent" in headers


class TestPubChemRetryEnvironmentVariable:
    """Test PubChem retry configuration via environment variable."""

    def test_default_retry_count(self):
        """Should use default retry count when env var not set."""
        from core.small_molecule_prep import _pubchem_get_json

        # Clear any existing env var
        env_backup = os.environ.get("PUBCHEM_MAX_RETRIES")
        if "PUBCHEM_MAX_RETRIES" in os.environ:
            del os.environ["PUBCHEM_MAX_RETRIES"]

        try:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"CID": 2244}

            with patch("requests.get", return_value=mock_response) as mock_get:
                _pubchem_get_json("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/json")

            # Should succeed on first try (default behavior)
            assert mock_get.call_count == 1
        finally:
            if env_backup is not None:
                os.environ["PUBCHEM_MAX_RETRIES"] = env_backup

    def test_custom_retry_count_from_env(self):
        """Should use custom retry count from environment variable."""
        import os
        from core.small_molecule_prep import _pubchem_get_json

        os.environ["PUBCHEM_MAX_RETRIES"] = "5"

        try:
            call_count = 0

            def mock_get(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 5:
                    raise requests.exceptions.ConnectionError("Error")
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"CID": 2244}
                return mock_response

            with patch("requests.get", side_effect=mock_get):
                result = _pubchem_get_json("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/json")

            assert call_count == 5
            assert result["CID"] == 2244
        finally:
            del os.environ["PUBCHEM_MAX_RETRIES"]


# Import os for env var tests
import os