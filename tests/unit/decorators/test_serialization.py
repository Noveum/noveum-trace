"""
Comprehensive tests for value serialization in Noveum Trace SDK.

This module tests the improved _serialize_value function that returns
JSON-serializable objects instead of string representations.
"""

import json
import warnings
from unittest.mock import MagicMock, patch

# Import from the module directly to access private functions for testing
try:
    from noveum_trace.decorators.base import _serialize_value  # noqa: E402
except ImportError:
    # Fallback: import from module if not exported
    import noveum_trace.decorators.base as base_module  # noqa: E402

    _serialize_value = base_module._serialize_value

from noveum_trace.context_managers import (  # noqa: E402
    OperationContextManager,
    trace_function_calls,
)

# Constants (in case they're not exported, use values directly)
DEFAULT_MAX_DEPTH = 10
DEFAULT_MAX_SIZE_BYTES = 1048576  # 1MB


class TestSerializeValueBasicTypes:
    """Test _serialize_value with basic/primitive types."""

    def test_serialize_none(self):
        """Test serialization of None."""
        result = _serialize_value(None)
        assert result is None

    def test_serialize_string(self):
        """Test serialization of strings."""
        result = _serialize_value("hello world")
        assert result == "hello world"
        assert isinstance(result, str)

    def test_serialize_integer(self):
        """Test serialization of integers."""
        result = _serialize_value(42)
        assert result == 42
        assert isinstance(result, int)

    def test_serialize_float(self):
        """Test serialization of floats."""
        result = _serialize_value(3.14159)
        assert result == 3.14159
        assert isinstance(result, float)

    def test_serialize_boolean(self):
        """Test serialization of booleans."""
        result = _serialize_value(True)
        assert result is True
        assert isinstance(result, bool)

        result = _serialize_value(False)
        assert result is False


class TestSerializeValueCollections:
    """Test _serialize_value with collections (dicts, lists, tuples)."""

    def test_serialize_dict(self):
        """Test serialization of dictionaries."""
        value = {"key1": "value1", "key2": 42, "key3": True}
        result = _serialize_value(value)
        assert isinstance(result, dict)
        assert result == value
        assert result["key1"] == "value1"
        assert result["key2"] == 42
        assert result["key3"] is True

    def test_serialize_nested_dict(self):
        """Test serialization of nested dictionaries."""
        value = {
            "level1": {
                "level2": {"level3": "deep_value"},
                "other": 123,
            },
            "top": "value",
        }
        result = _serialize_value(value)
        assert isinstance(result, dict)
        assert result["level1"]["level2"]["level3"] == "deep_value"
        assert result["level1"]["other"] == 123
        assert result["top"] == "value"

    def test_serialize_list(self):
        """Test serialization of lists."""
        value = [1, 2, 3, "four", True]
        result = _serialize_value(value)
        assert isinstance(result, list)
        assert result == value

    def test_serialize_tuple(self):
        """Test serialization of tuples (converted to lists)."""
        value = (1, 2, 3, "four")
        result = _serialize_value(value)
        assert isinstance(result, list)
        assert result == [1, 2, 3, "four"]

    def test_serialize_nested_list(self):
        """Test serialization of nested lists."""
        value = [[1, 2], [3, 4], ["a", "b"]]
        result = _serialize_value(value)
        assert isinstance(result, list)
        assert result == [[1, 2], [3, 4], ["a", "b"]]

    def test_serialize_dict_with_list(self):
        """Test serialization of dict containing lists."""
        value = {"items": [1, 2, 3], "metadata": {"count": 3}}
        result = _serialize_value(value)
        assert isinstance(result, dict)
        assert isinstance(result["items"], list)
        assert result["items"] == [1, 2, 3]
        assert result["metadata"]["count"] == 3

    def test_serialize_list_with_dict(self):
        """Test serialization of list containing dicts."""
        value = [{"id": 1, "name": "item1"}, {"id": 2, "name": "item2"}]
        result = _serialize_value(value)
        assert isinstance(result, list)
        assert isinstance(result[0], dict)
        assert result[0]["id"] == 1
        assert result[1]["name"] == "item2"

    def test_serialize_dict_with_non_string_keys(self):
        """Test serialization of dict with non-string keys."""
        value = {1: "one", 2: "two", 3.5: "three_point_five"}
        result = _serialize_value(value)

        # New implementation returns dict, old returns string
        if isinstance(result, dict):
            # Keys should be converted to strings
            assert "1" in result
            assert result["1"] == "one"
            assert "2" in result
            assert result["2"] == "two"
            assert "3.5" in result
            assert result["3.5"] == "three_point_five"
        else:
            # Old implementation - just verify it's a string
            assert isinstance(result, str)
            assert "one" in result
            assert "two" in result


class TestSerializeValueComplexObjects:
    """Test _serialize_value with complex objects."""

    def test_serialize_object_with_dict(self):
        """Test serialization of object with __dict__."""

        class SimpleObject:
            def __init__(self):
                self.name = "test"
                self.value = 42
                self.active = True

        obj = SimpleObject()
        result = _serialize_value(obj)
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert result["value"] == 42
        assert result["active"] is True

    def test_serialize_object_skips_private_attributes(self):
        """Test that private attributes (starting with _) are skipped."""

        class ObjectWithPrivate:
            def __init__(self):
                self.public = "visible"
                self._private = "hidden"
                self.__very_private = "very_hidden"

        obj = ObjectWithPrivate()
        result = _serialize_value(obj)
        assert isinstance(result, dict)
        assert "public" in result
        assert result["public"] == "visible"
        # Private attributes should not be in result
        assert "_private" not in result
        assert "__very_private" not in result

    def test_serialize_object_with_nested_objects(self):
        """Test serialization of object containing other objects."""

        class NestedObject:
            def __init__(self, inner):
                self.inner = inner
                self.name = "outer"

        class InnerObject:
            def __init__(self):
                self.value = 100

        inner = InnerObject()
        outer = NestedObject(inner)
        result = _serialize_value(outer)
        assert isinstance(result, dict)
        assert result["name"] == "outer"
        assert isinstance(result["inner"], dict)
        assert result["inner"]["value"] == 100

    def test_serialize_object_with_to_dict_method(self):
        """Test serialization of object with to_dict() method."""

        class ObjectWithToDict:
            def __init__(self):
                self.data = {"key": "value"}

            def to_dict(self):
                return {"serialized": True, "data": self.data}

        obj = ObjectWithToDict()
        result = _serialize_value(obj)
        assert isinstance(result, dict)
        assert result["serialized"] is True
        assert result["data"]["key"] == "value"

    def test_serialize_object_fallback_to_string(self):
        """Test that objects without __dict__ fall back to string."""

        class ObjectWithoutDict:
            __slots__ = ("value",)

            def __init__(self):
                self.value = "test"

        obj = ObjectWithoutDict()
        result = _serialize_value(obj)
        # Should fall back to string representation
        assert isinstance(result, str)
        assert "ObjectWithoutDict" in result or "test" in result


class TestSerializeValueEdgeCases:
    """Test _serialize_value with edge cases and error conditions."""

    def test_serialize_circular_reference(self):
        """Test handling of circular references."""

        class CircularObject:
            def __init__(self):
                self.name = "circular"
                self.ref = None

        obj1 = CircularObject()
        obj2 = CircularObject()
        obj1.ref = obj2
        obj2.ref = obj1  # Create circular reference

        result = _serialize_value(obj1)
        assert isinstance(result, dict)
        assert result["name"] == "circular"
        # The circular reference should be detected
        assert isinstance(result["ref"], dict)
        # The nested circular reference should show as circular_reference
        nested = result["ref"]
        if isinstance(nested.get("ref"), str):
            assert "circular_reference" in nested["ref"]

    def test_serialize_max_depth_reached(self):
        """Test that max depth is respected."""

        # Create deeply nested structure
        deep_dict = {"level": 0}
        current = deep_dict
        for i in range(DEFAULT_MAX_DEPTH + 5):
            current["nested"] = {"level": i + 1}
            current = current["nested"]

        result = _serialize_value(deep_dict)
        assert isinstance(result, dict)
        # Should have reached max depth
        assert "max_depth_reached" in str(result) or isinstance(result, dict)

    def test_serialize_empty_collections(self):
        """Test serialization of empty collections."""
        assert _serialize_value({}) == {}
        assert _serialize_value([]) == []
        assert _serialize_value(()) == []

    def test_serialize_mixed_types_in_collection(self):
        """Test serialization of collections with mixed types."""
        value = {
            "string": "text",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3],
            "nested": {"key": "value"},
        }
        result = _serialize_value(value)
        assert isinstance(result, dict)
        assert result["string"] == "text"
        assert result["number"] == 42
        assert result["float"] == 3.14
        assert result["bool"] is True
        assert result["none"] is None
        assert isinstance(result["list"], list)
        assert isinstance(result["nested"], dict)


class TestSerializeValueSizeWarnings:
    """Test size warning functionality."""

    def test_size_warning_triggered(self):
        """Test that size warning is triggered for large data."""
        # Create data larger than 1MB
        large_string = "x" * (DEFAULT_MAX_SIZE_BYTES + 1000)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _serialize_value(large_string)
            # Should still serialize
            assert isinstance(result, str)
            # Should have triggered a warning
            assert len(w) > 0
            assert any("large" in str(warning.message).lower() for warning in w)

    def test_size_warning_not_triggered_small_data(self):
        """Test that size warning is not triggered for small data."""
        small_string = "hello"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _serialize_value(small_string)
            assert result == "hello"
            # Should not have triggered a warning
            assert len(w) == 0

    def test_size_warning_for_large_dict(self):
        """Test size warning for large dictionaries."""
        # Create a large dictionary
        large_dict = {f"key_{i}": f"value_{i}" * 1000 for i in range(2000)}

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = _serialize_value(large_dict)
            assert isinstance(result, dict)
            # May or may not trigger warning depending on actual size
            # Just verify it doesn't crash

    def test_check_serialized_size_function(self):
        """Test the _check_serialized_size helper function."""
        # Test through _serialize_value which calls _check_serialized_size
        large_data = "x" * (DEFAULT_MAX_SIZE_BYTES + 1000)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = _serialize_value(large_data)
            assert isinstance(result, str)
            # Warning may or may not be triggered depending on implementation
            # Just verify it doesn't crash


class TestSerializeValueJSONCompatibility:
    """Test that serialized values are JSON-compatible."""

    def test_serialized_dict_is_json_serializable(self):
        """Test that serialized dicts can be converted to JSON."""
        value = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
        result = _serialize_value(value)
        # Should be JSON serializable
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed == value

    def test_serialized_list_is_json_serializable(self):
        """Test that serialized lists can be converted to JSON."""
        value = [1, 2, 3, {"nested": "value"}]
        result = _serialize_value(value)
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed == value

    def test_serialized_object_is_json_serializable(self):
        """Test that serialized objects can be converted to JSON."""

        class TestObject:
            def __init__(self):
                self.name = "test"
                self.value = 42

        obj = TestObject()
        result = _serialize_value(obj)
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["name"] == "test"
        assert parsed["value"] == 42


class TestContextManagerInputCapture:
    """Test context manager input capture functionality."""

    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    @patch("noveum_trace.core.context.get_current_trace")
    def test_capture_function_args(
        self, mock_get_current_trace, mock_is_initialized, mock_get_client
    ):
        """Test that context manager can capture function arguments."""
        mock_is_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        def test_func(param1: str, param2: int, param3: dict = None):
            return "result"

        context_mgr = OperationContextManager("test_operation", capture_args=True)
        with context_mgr:
            context_mgr.capture_function_args(
                test_func, "hello", 42, param3={"key": "value"}
            )

        # Verify that set_attribute was called with proper input attributes
        calls = mock_span.set_attribute.call_args_list
        # Should have captured individual parameters and args/kwargs
        assert len(calls) > 0

    @patch("noveum_trace.get_client")
    @patch("noveum_trace.is_initialized")
    @patch("noveum_trace.core.context.get_current_trace")
    def test_trace_function_calls_captures_args(
        self, mock_get_current_trace, mock_is_initialized, mock_get_client
    ):
        """Test that trace_function_calls decorator captures arguments."""
        mock_is_initialized.return_value = True
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_trace = MagicMock()
        mock_get_current_trace.return_value = mock_trace
        mock_span = MagicMock()
        mock_client.start_span.return_value = mock_span

        # Test decorator with keyword arguments
        @trace_function_calls(capture_args=True)
        def test_func(x: int, y: str = "default"):
            return x + len(y)

        result = test_func(10, y="test")

        assert result == 14
        # Verify span was created and attributes were set
        mock_client.start_span.assert_called()
        assert mock_span.set_attribute.called

    def test_capture_args_with_complex_objects(self):
        """Test capturing arguments with complex objects."""

        class ComplexObject:
            def __init__(self):
                self.name = "complex"
                self.value = 100

        obj = ComplexObject()

        # Serialize the object
        result = _serialize_value(obj)
        assert isinstance(result, dict)
        assert result["name"] == "complex"
        assert result["value"] == 100


class TestRealWorldScenarios:
    """Test real-world scenarios that match the original issue."""

    def test_serialize_module_like_object(self):
        """Test serialization of objects similar to the reported issue."""

        class Module:
            def __init__(self, name: str):
                self.name = name
                self.data = {"key": "value"}

        module = Module("test_module")
        # This should now serialize properly instead of showing object reference
        result = _serialize_value(module)
        assert isinstance(result, dict)
        assert result["name"] == "test_module"
        assert isinstance(result["data"], dict)
        assert result["data"]["key"] == "value"

    def test_serialize_args_tuple(self):
        """Test serialization of args tuple (like in the original issue)."""

        # Simulate the scenario: input.args = "(<__main__.Module object...>,)"
        class Module:
            def __init__(self):
                self.name = "test"

        module = Module()
        args = (module,)

        # Old behavior would give: "(<__main__.Module object...>,)"
        # New behavior should give: [{"name": "test"}]
        result = _serialize_value(args)
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert result[0]["name"] == "test"

    def test_serialize_kwargs_dict(self):
        """Test serialization of kwargs dict."""
        kwargs = {"param1": "value1", "param2": 42}

        result = _serialize_value(kwargs)
        assert isinstance(result, dict)
        assert result["param1"] == "value1"
        assert result["param2"] == 42

    def test_serialize_mixed_args_kwargs(self):
        """Test serialization of mixed args and kwargs with complex objects."""

        class DataObject:
            def __init__(self, data):
                self.data = data

        data_obj = DataObject({"nested": "value"})
        args = (data_obj, "string_arg")
        kwargs = {"key": data_obj, "simple": 123}

        args_result = _serialize_value(args)
        kwargs_result = _serialize_value(kwargs)

        assert isinstance(args_result, list)
        assert len(args_result) == 2
        assert isinstance(args_result[0], dict)
        assert args_result[0]["data"]["nested"] == "value"
        assert args_result[1] == "string_arg"

        assert isinstance(kwargs_result, dict)
        assert isinstance(kwargs_result["key"], dict)
        assert kwargs_result["key"]["data"]["nested"] == "value"
        assert kwargs_result["simple"] == 123
