"""Unit tests for the retrieval decorator functionality."""

from unittest.mock import Mock, patch

import pytest

from noveum_trace.core.span import SpanStatus
from noveum_trace.decorators.retrieval import (
    _extract_metadata_from_list,
    _extract_query,
    _extract_retrieval_results,
    _extract_scores_from_list,
    _get_result_count,
    trace_retrieval,
)


class TestRetrievalDecorator:
    """Test suite for retrieval decorator functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        # Mock the global client
        self.mock_client = Mock()
        self.mock_span = Mock()
        self.mock_span.set_attributes = Mock()
        self.mock_span.set_attribute = Mock()
        self.mock_span.set_status = Mock()
        self.mock_span.record_exception = Mock()
        self.mock_span.add_tags = Mock()
        self.mock_span.__enter__ = Mock(return_value=self.mock_span)
        self.mock_span.__exit__ = Mock(return_value=None)
        self.mock_client.start_span.return_value = self.mock_span

    def test_trace_retrieval_basic(self):
        """Test basic retrieval tracing."""

        @trace_retrieval(index_name="test_index")
        def search_function(query: str):
            return [{"id": 1, "text": "Result 1", "score": 0.9}]

        with patch("noveum_trace.get_client", return_value=self.mock_client):
            with patch("noveum_trace.is_initialized", return_value=True):
                result = search_function("test query")

        assert result == [{"id": 1, "text": "Result 1", "score": 0.9}]
        self.mock_client.start_span.assert_called_once()

        # Check span was used properly
        self.mock_span.set_attributes.assert_called()
        self.mock_span.set_status.assert_called()

    @pytest.mark.parametrize(
        "retrieval_type,expected",
        [
            ("vector_search", "vector_search"),
            ("keyword_search", "keyword_search"),
            ("hybrid_search", "hybrid_search"),
            ("semantic_search", "semantic_search"),
        ],
    )
    def test_trace_retrieval_types(self, retrieval_type, expected):
        """Test different retrieval types."""

        @trace_retrieval(retrieval_type=retrieval_type)
        def search_function(query: str):
            return []

        with patch("noveum_trace.get_client", return_value=self.mock_client):
            with patch("noveum_trace.is_initialized", return_value=True):
                search_function("test")

        # Verify function was called and span was created
        self.mock_client.start_span.assert_called_once()
        self.mock_span.set_attributes.assert_called()

    def test_trace_retrieval_with_custom_name(self):
        """Test retrieval tracing with custom span name."""

        @trace_retrieval(name="CustomSearch")
        def search_function(query: str):
            return []

        with patch("noveum_trace.get_client", return_value=self.mock_client):
            with patch("noveum_trace.is_initialized", return_value=True):
                search_function("test")

        # Verify span was created with custom name
        self.mock_client.start_span.assert_called_once()
        call_args = self.mock_client.start_span.call_args
        assert "CustomSearch" in call_args.kwargs["name"]

    def test_trace_retrieval_with_metadata_and_tags(self):
        """Test retrieval tracing with metadata and tags."""
        metadata = {"model": "embedding-v1", "dimension": 768}
        tags = {"team": "search", "version": "2.0"}

        @trace_retrieval(metadata=metadata, tags=tags)
        def search_function(query: str):
            return []

        with patch("noveum_trace.get_client", return_value=self.mock_client):
            with patch("noveum_trace.is_initialized", return_value=True):
                search_function("test")

        # Verify tracing occurred with metadata and tags
        self.mock_client.start_span.assert_called_once()
        self.mock_span.set_attributes.assert_called()

    def test_trace_retrieval_capture_flags(self):
        """Test retrieval tracing with different capture flags."""
        results = [
            {"id": 1, "text": "Result 1", "score": 0.9, "metadata": {"source": "doc1"}},
            {"id": 2, "text": "Result 2", "score": 0.8, "metadata": {"source": "doc2"}},
        ]

        @trace_retrieval(
            capture_query=False,
            capture_results=False,
            capture_scores=True,
            capture_metadata=True,
        )
        def search_function(query: str):
            return results

        with patch("noveum_trace.get_client", return_value=self.mock_client):
            with patch("noveum_trace.is_initialized", return_value=True):
                search_function("test query")

        # Verify tracing occurred
        self.mock_client.start_span.assert_called_once()
        self.mock_span.set_attributes.assert_called()

    def test_trace_retrieval_max_results(self):
        """Test retrieval tracing with max_results limit."""
        # Create 100 results
        results = [
            {"id": i, "text": f"Result {i}", "score": 0.9 - i * 0.001}
            for i in range(100)
        ]

        @trace_retrieval(max_results=10, capture_results=True)
        def search_function(query: str):
            return results

        with patch("noveum_trace.get_client", return_value=self.mock_client):
            with patch("noveum_trace.is_initialized", return_value=True):
                search_function("test")

        # Verify tracing occurred with large dataset
        self.mock_client.start_span.assert_called_once()
        self.mock_span.set_attributes.assert_called()

    def test_trace_retrieval_exception_handling(self):
        """Test retrieval tracing with exception handling."""

        @trace_retrieval()
        def failing_search(query: str):
            raise ValueError("Search failed")

        with patch("noveum_trace.get_client", return_value=self.mock_client):
            with patch("noveum_trace.is_initialized", return_value=True):
                with pytest.raises(ValueError):
                    failing_search("test")

        self.mock_span.record_exception.assert_called_once()
        self.mock_span.set_status.assert_called_with(SpanStatus.ERROR, "Search failed")

    def test_trace_retrieval_direct_decoration(self):
        """Test direct function decoration without parentheses."""

        @trace_retrieval
        def search_function(query: str):
            return ["result1", "result2"]

        with patch("noveum_trace.get_client", return_value=self.mock_client):
            with patch("noveum_trace.is_initialized", return_value=True):
                result = search_function("test")

        assert result == ["result1", "result2"]
        self.mock_client.start_span.assert_called_once()

    def test_trace_retrieval_async_function(self):
        """Test retrieval tracing with async functions."""

        @trace_retrieval()
        async def async_search(query: str):
            return [{"id": 1, "score": 0.9}]

        import asyncio

        with patch("noveum_trace.get_client", return_value=self.mock_client):
            with patch("noveum_trace.is_initialized", return_value=True):
                result = asyncio.run(async_search("test"))

        assert result == [{"id": 1, "score": 0.9}]
        self.mock_client.start_span.assert_called_once()


class TestRetrievalHelperFunctions:
    """Test suite for retrieval helper functions."""

    def test_extract_query_various_params(self):
        """Test extracting query from various parameter names."""
        # Test with 'query' parameter
        result = _extract_query({"query": "search term"})
        assert result == "search term"

        # Test with 'search_query' parameter
        result = _extract_query({"search_query": "another term"})
        assert result == "another term"

        # Test with 'text' parameter
        result = _extract_query({"text": "text search"})
        assert result == "text search"

        # Test with 'question' parameter
        result = _extract_query({"question": "what is this?"})
        assert result == "what is this?"

        # Test with 'prompt' parameter
        result = _extract_query({"prompt": "generate this"})
        assert result == "generate this"

    def test_extract_query_case_insensitive(self):
        """Test that query extraction is case insensitive."""
        # The function converts parameter names to lowercase
        result = _extract_query({"Query": "uppercase query"})
        assert result == "uppercase query"

        result = _extract_query({"SEARCH_QUERY": "all caps"})
        assert result == "all caps"

    def test_extract_query_none(self):
        """Test extracting query when not found."""
        result = _extract_query({"other_param": "value"})
        assert result is None

        result = _extract_query({})
        assert result is None

    def test_get_result_count_various_types(self):
        """Test getting result count from various result types."""
        # List
        assert _get_result_count([1, 2, 3]) == 3

        # Tuple
        assert _get_result_count((1, 2, 3, 4)) == 4

        # Dict with 'results' key
        assert _get_result_count({"results": [1, 2, 3, 4, 5]}) == 5

        # Dict without 'results' key (falls back to __len__)
        assert _get_result_count({"hits": [1, 2], "total": 2}) == 2

        # String (has __len__)
        assert _get_result_count("string") == 6

        # Number (no __len__)
        assert _get_result_count(42) == 1

        # None
        assert _get_result_count(None) == 0

    @pytest.mark.parametrize(
        "results,expected_scores",
        [
            # List of dicts with score
            ([{"score": 0.9}, {"score": 0.8}, {"score": 0.7}], [0.9, 0.8, 0.7]),
            # List of dicts with similarity (actual field name in function)
            ([{"similarity": 0.99}, {"similarity": 0.89}], [0.99, 0.89]),
            # List of dicts with relevance
            ([{"relevance": 0.95}, {"relevance": 0.85}], [0.95, 0.85]),
            # List of tuples (data, score)
            ([("data1", 0.9), ("data2", 0.8)], [0.9, 0.8]),
            # Mixed types
            ([{"score": 0.9}, {"no_score": "data"}, {"score": 0.7}], [0.9, 0.7]),
            # Empty list
            ([], None),
            # No valid scores
            ([{"no_score": "data"}], None),
        ],
    )
    def test_extract_scores_from_list(self, results, expected_scores):
        """Test extracting scores from various list formats."""
        scores = _extract_scores_from_list(results)
        assert scores == expected_scores

    def test_extract_metadata_from_list(self):
        """Test extracting metadata from list of results."""
        results = [
            {"id": 1, "source": "doc1", "document_id": "d1"},
            {"id": 2, "source": "doc2", "chunk_id": "c1"},
            {"id": 3},  # No metadata
            {"id": 4, "metadata_type": "text"},  # Key starts with metadata
        ]

        metadata_list = _extract_metadata_from_list(results)

        assert len(metadata_list) == 3
        assert metadata_list[0] == {"source": "doc1", "document_id": "d1"}
        assert metadata_list[1] == {"source": "doc2", "chunk_id": "c1"}
        assert metadata_list[2] == {"metadata_type": "text"}

    def test_extract_retrieval_results_comprehensive(self):
        """Test comprehensive extraction of retrieval results."""
        # Test with list of results
        results = [
            {"id": 1, "text": "Result 1", "score": 0.9, "metadata": {"source": "doc1"}},
            {"id": 2, "text": "Result 2", "score": 0.8, "metadata": {"source": "doc2"}},
            {"id": 3, "text": "Result 3", "score": 0.7, "metadata": {"source": "doc3"}},
        ]

        extracted = _extract_retrieval_results(
            results,
            capture_results=True,
            capture_scores=True,
            capture_metadata=True,
            max_results=2,
        )

        assert extracted["retrieval.result_count"] == 3
        assert extracted["retrieval.result_type"] == "list"
        assert len(extracted["retrieval.sample_results"]) == 2
        assert extracted["retrieval.results_truncated"] is True
        assert extracted["retrieval.total_results"] == 3
        assert extracted["retrieval.scores"] == [0.9, 0.8, 0.7]
        assert extracted["retrieval.max_score"] == 0.9
        assert extracted["retrieval.min_score"] == 0.7
        assert abs(extracted["retrieval.avg_score"] - 0.8) < 0.001
        assert len(extracted["retrieval.result_metadata"]) == 3

    def test_extract_retrieval_results_dict_format(self):
        """Test extraction from dictionary format results."""
        results = {
            "results": [
                {"id": 1, "text": "Result 1"},
                {"id": 2, "text": "Result 2"},
                {"id": 3, "text": "Result 3"},
            ],
            "scores": [0.95, 0.85, 0.75],
            "total": 3,
            "query_time": 0.123,
        }

        extracted = _extract_retrieval_results(
            results, capture_results=True, capture_scores=True, max_results=2
        )

        assert extracted["retrieval.result_count"] == 3
        assert extracted["retrieval.result_type"] == "dict"
        assert len(extracted["retrieval.sample_results"]) == 2
        assert extracted["retrieval.scores"] == [0.95, 0.85, 0.75]
        assert extracted["retrieval.max_score"] == 0.95
        assert extracted["retrieval.min_score"] == 0.75

    def test_extract_retrieval_results_no_capture(self):
        """Test extraction with all capture flags set to False."""
        results = [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.8}]

        extracted = _extract_retrieval_results(
            results, capture_results=False, capture_scores=False, capture_metadata=False
        )

        assert extracted["retrieval.result_count"] == 2
        assert extracted["retrieval.result_type"] == "list"
        assert "retrieval.sample_results" not in extracted
        assert "retrieval.scores" not in extracted
        assert "retrieval.result_metadata" not in extracted

    def test_extract_retrieval_results_edge_cases(self):
        """Test extraction with edge cases."""
        # Empty results
        extracted = _extract_retrieval_results([])
        assert extracted["retrieval.result_count"] == 0
        assert extracted["retrieval.result_type"] == "list"

        # None results
        extracted = _extract_retrieval_results(None)
        assert extracted["retrieval.result_count"] == 0
        assert extracted["retrieval.result_type"] == "NoneType"

        # String results (has length)
        extracted = _extract_retrieval_results("single result")
        assert extracted["retrieval.result_count"] == 13  # Length of string
        assert extracted["retrieval.result_type"] == "str"
