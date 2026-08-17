"""
Tests for Priority 8: Method-specific cache statistics.

Verifies:
- Cache statistics are reported per method (HDOCK, Vina, etc.)
- Per-method hit/miss counts are tracked
- Per-method byte sizes are tracked
- Total statistics are computed correctly
"""

import pytest


class TestMethodSpecificCacheStatistics:
    """Test method-specific cache statistics."""

    def test_cache_stats_tracked_per_method(self):
        """Cache statistics should be tracked per method."""
        from orchestration.cache_reporter import CacheReporter

        reporter = CacheReporter()

        # Record hits for different methods
        reporter.record_hit("HDOCK", size_bytes=1000)
        reporter.record_hit("Vina", size_bytes=2000)
        reporter.record_hit("HDOCK", size_bytes=1500)

        stats = reporter.get_statistics()

        assert "HDOCK" in stats["by_method"]
        assert "Vina" in stats["by_method"]
        assert stats["by_method"]["HDOCK"]["hits"] == 2
        assert stats["by_method"]["Vina"]["hits"] == 1

    def test_hit_miss_counts_per_method(self):
        """Hit and miss counts should be tracked per method."""
        from orchestration.cache_reporter import CacheReporter

        reporter = CacheReporter()

        reporter.record_hit("HDOCK", size_bytes=1000)
        reporter.record_hit("HDOCK", size_bytes=1000)
        reporter.record_miss("HDOCK")
        reporter.record_hit("Vina", size_bytes=2000)

        stats = reporter.get_statistics()

        assert stats["by_method"]["HDOCK"]["hits"] == 2
        assert stats["by_method"]["HDOCK"]["misses"] == 1
        assert stats["by_method"]["Vina"]["hits"] == 1
        assert stats["by_method"]["Vina"]["misses"] == 0

    def test_byte_sizes_tracked_per_method(self):
        """Byte sizes should be tracked per method."""
        from orchestration.cache_reporter import CacheReporter

        reporter = CacheReporter()

        reporter.record_hit("HDOCK", size_bytes=1000)
        reporter.record_hit("HDOCK", size_bytes=2000)
        reporter.record_hit("Vina", size_bytes=500)

        stats = reporter.get_statistics()

        assert stats["by_method"]["HDOCK"]["total_bytes"] == 3000
        assert stats["by_method"]["Vina"]["total_bytes"] == 500

    def test_total_statistics_computed(self):
        """Total statistics should be computed from per-method stats."""
        from orchestration.cache_reporter import CacheReporter

        reporter = CacheReporter()

        reporter.record_hit("HDOCK", size_bytes=1000)
        reporter.record_hit("Vina", size_bytes=2000)
        reporter.record_miss("HDOCK")

        stats = reporter.get_statistics()

        assert stats["total"]["hits"] == 2
        assert stats["total"]["misses"] == 1
        assert stats["total"]["total_bytes"] == 3000

    def test_hit_rate_computed_per_method(self):
        """Hit rate should be computed per method."""
        from orchestration.cache_reporter import CacheReporter

        reporter = CacheReporter()

        reporter.record_hit("HDOCK", size_bytes=1000)
        reporter.record_hit("HDOCK", size_bytes=1000)
        reporter.record_miss("HDOCK")
        reporter.record_miss("HDOCK")

        stats = reporter.get_statistics()

        # 2 hits, 2 misses = 50% hit rate
        assert stats["by_method"]["HDOCK"]["hit_rate"] == 0.5

    def test_overall_hit_rate_computed(self):
        """Overall hit rate should be computed."""
        from orchestration.cache_reporter import CacheReporter

        reporter = CacheReporter()

        reporter.record_hit("HDOCK", size_bytes=1000)
        reporter.record_hit("Vina", size_bytes=1000)
        reporter.record_miss("HDOCK")
        reporter.record_miss("Vina")

        stats = reporter.get_statistics()

        # 2 hits, 2 misses = 50% hit rate
        assert stats["total"]["hit_rate"] == 0.5

    def test_unknown_method_handled(self):
        """Unknown method should be handled gracefully."""
        from orchestration.cache_reporter import CacheReporter

        reporter = CacheReporter()

        reporter.record_hit("UnknownMethod", size_bytes=1000)

        stats = reporter.get_statistics()

        assert "UnknownMethod" in stats["by_method"]


class TestCacheStatisticsPersistence:
    """Test cache statistics persistence."""

    def test_statistics_summarized(self):
        """Statistics should include a summary."""
        from orchestration.cache_reporter import CacheReporter

        reporter = CacheReporter()

        reporter.record_hit("HDOCK", size_bytes=1000)
        reporter.record_hit("Vina", size_bytes=2000)

        stats = reporter.get_statistics()

        assert "summary" in stats
        assert "total_requests" in stats["summary"]

    def test_statistics_reset(self):
        """Statistics should be resettable."""
        from orchestration.cache_reporter import CacheReporter

        reporter = CacheReporter()

        reporter.record_hit("HDOCK", size_bytes=1000)
        reporter.reset()

        stats = reporter.get_statistics()

        assert stats["total"]["hits"] == 0
        assert stats["total"]["misses"] == 0


class TestCacheStatisticsReporting:
    """Test cache statistics reporting format."""

    def test_report_is_dict(self):
        """Report should be a dictionary."""
        from orchestration.cache_reporter import CacheReporter

        reporter = CacheReporter()

        stats = reporter.get_statistics()

        assert isinstance(stats, dict)

    def test_report_has_by_method(self):
        """Report should have by_method section."""
        from orchestration.cache_reporter import CacheReporter

        reporter = CacheReporter()

        stats = reporter.get_statistics()

        assert "by_method" in stats

    def test_report_has_total(self):
        """Report should have total section."""
        from orchestration.cache_reporter import CacheReporter

        reporter = CacheReporter()

        stats = reporter.get_statistics()

        assert "total" in stats