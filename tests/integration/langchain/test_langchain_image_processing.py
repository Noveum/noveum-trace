"""Behavioral tests for the LangChain integration's image pipeline.

These exercise the *real* image-processing code in
``noveum_trace.integrations.langchain.message_utils`` (base64 + http branches,
the JSON-prompt gate, and the format-detection cascade in
``download_image_from_url``) plus the integration-level payoff: real captured
``llm.input.image_count`` / ``llm.input.image_uuids`` span attributes produced
by ``on_chat_model_start``.

The autouse conftest fixtures block all real HTTP, so the http branch is
covered by patching the local ``requests.get`` / ``download_image_from_url``
seams.  Base64 data URIs need no network.
"""

from __future__ import annotations

import base64
import io
import json
from unittest.mock import MagicMock, patch

import pytest

from ._helpers import (
    LANGCHAIN_AVAILABLE,
    find_span,
    get_exported_spans,
)

pytestmark = pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE, reason="LangChain not available"
)


def _png_bytes() -> bytes:
    """Return bytes of a tiny but real PNG (PIL can detect its format)."""
    from PIL import Image

    img = Image.new("RGB", (2, 2), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _png_data_uri() -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes()).decode()


def test_extract_and_process_base64_image_uploads_and_replaces(
    client_with_mocked_transport,
):
    """Base64 data:image path: upload once, replace url with image://uuid,
    and never mutate the caller's input (deepcopy guarantee)."""
    from noveum_trace.integrations.langchain.message_utils import (
        extract_and_process_images,
    )
    from noveum_trace.utils.image_utils import format_image_reference

    client = client_with_mocked_transport
    data_uri = _png_data_uri()

    # One batch, one message, multimodal content with a single base64 image.
    image_url_dict = {"url": data_uri}
    msg = {
        "type": "human",
        "content": [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": image_url_dict},
        ],
    }
    message_dicts = [[msg]]

    uuids, modified = extract_and_process_images(
        message_dicts, trace_id="trace-1", span_id="span-1"
    )

    # Exactly one uuid extracted.
    assert isinstance(uuids, list)
    assert len(uuids) == 1
    extracted_uuid = uuids[0]

    # The returned dict's url is replaced with the image:// reference.
    new_url = modified[0][0]["content"][1]["image_url"]["url"]
    assert new_url == format_image_reference(extracted_uuid)
    assert new_url == f"image://{extracted_uuid}"

    # The image was uploaded exactly once with the decoded bytes + metadata.
    assert client.transport.export_image.call_count == 1
    kwargs = client.transport.export_image.call_args.kwargs
    assert kwargs["image_data"] == _png_bytes()
    assert kwargs["trace_id"] == "trace-1"
    assert kwargs["span_id"] == "span-1"
    assert kwargs["image_uuid"] == extracted_uuid
    assert kwargs["metadata"] == {"format": "png"}

    # Deepcopy guarantee: the ORIGINAL input dict is untouched.
    assert image_url_dict["url"] == data_uri
    assert msg["content"][1]["image_url"]["url"] == data_uri


def test_extract_and_process_http_image_download_mocked(client_with_mocked_transport):
    """http(s) image branch both ways: a successful download is uploaded
    (metadata carries original_url) and the url replaced; a failed download
    (returns None) leaves the url unchanged with no upload."""
    from noveum_trace.integrations.langchain import message_utils
    from noveum_trace.utils.image_utils import format_image_reference

    client = client_with_mocked_transport
    http_url = "https://example.com/cat.png"

    def _make_messages():
        return [
            [
                {
                    "type": "human",
                    "content": [
                        {"type": "image_url", "image_url": {"url": http_url}},
                    ],
                }
            ]
        ]

    # --- success branch: download returns (bytes, "png") ---
    with patch.object(
        message_utils,
        "download_image_from_url",
        return_value=(b"rawbytes", "png"),
    ) as dl:
        uuids, modified = message_utils.extract_and_process_images(
            _make_messages(), trace_id="t", span_id="s"
        )

    dl.assert_called_once_with(http_url)
    assert len(uuids) == 1
    new_url = modified[0][0]["content"][0]["image_url"]["url"]
    assert new_url == format_image_reference(uuids[0])
    assert client.transport.export_image.call_count == 1
    kwargs = client.transport.export_image.call_args.kwargs
    assert kwargs["image_data"] == b"rawbytes"
    assert kwargs["metadata"] == {"format": "png", "original_url": http_url}

    # --- failure branch: download returns None ---
    client.transport.export_image.reset_mock()
    with patch.object(
        message_utils, "download_image_from_url", return_value=None
    ) as dl2:
        uuids2, modified2 = message_utils.extract_and_process_images(
            _make_messages(), trace_id="t", span_id="s"
        )

    dl2.assert_called_once_with(http_url)
    assert uuids2 == []
    # url left UNCHANGED, no upload.
    assert modified2[0][0]["content"][0]["image_url"]["url"] == http_url
    assert client.transport.export_image.call_count == 0


def test_process_images_from_prompts_substring_gate_and_replace(
    client_with_mocked_transport,
):
    """process_images_from_prompts: plain text passes through untouched (gate);
    a JSON-list prompt with an embedded base64 image is re-serialized with the
    image:// reference; a prompt that mentions the substring but doesn't start
    with [ or { is left unparsed."""
    from noveum_trace.integrations.langchain.message_utils import (
        process_images_from_prompts,
    )

    client = client_with_mocked_transport

    # 1) Plain text -> substring gate keeps it unchanged, no uploads.
    uuids, processed = process_images_from_prompts(
        ["just some plain text"], trace_id="t", span_id="s"
    )
    assert uuids == []
    assert processed == ["just some plain text"]
    assert client.transport.export_image.call_count == 0

    # 2) JSON-list prompt embedding a base64 image_url.  Built with json.dumps
    #    so the literal '"type": "image_url"' gate (exact spacing) trips.
    data_uri = _png_data_uri()
    json_prompt = json.dumps(
        [
            {
                "type": "human",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]
    )
    assert '"type": "image_url"' in json_prompt  # gate sanity check

    uuids2, processed2 = process_images_from_prompts(
        [json_prompt], trace_id="t", span_id="s"
    )
    assert len(uuids2) == 1
    assert client.transport.export_image.call_count == 1
    # Re-serialized: base64 gone, image:// reference present.
    assert data_uri not in processed2[0]
    assert f"image://{uuids2[0]}" in processed2[0]
    # Still valid JSON round-tripping to the replaced reference.
    reparsed = json.loads(processed2[0])
    assert reparsed[0]["content"][0]["image_url"]["url"] == f"image://{uuids2[0]}"

    # 3) Mentions the substring but does NOT start with [ or { -> not parsed,
    #    returned unchanged, no further uploads.
    client.transport.export_image.reset_mock()
    non_bracket = 'prefix "type": "image_url" trailing'
    uuids3, processed3 = process_images_from_prompts(
        [non_bracket], trace_id="t", span_id="s"
    )
    assert uuids3 == []
    assert processed3 == [non_bracket]
    assert client.transport.export_image.call_count == 0


def test_download_image_from_url_format_detection_and_failures():
    """download_image_from_url format-detection cascade and the
    always-returns-None-on-error contract, with requests.get patched."""
    from noveum_trace.integrations.langchain.message_utils import (
        download_image_from_url,
    )

    # 1) PIL-success path: real PNG bytes -> "png".
    resp_png = MagicMock()
    resp_png.content = _png_bytes()
    resp_png.headers = {}
    resp_png.raise_for_status = MagicMock()
    with patch("requests.get", return_value=resp_png):
        result = download_image_from_url("https://x/a.png")
    assert result is not None
    data, fmt = result
    assert data == _png_bytes()
    assert fmt == "png"

    # 2) PIL fails (non-image bytes), Content-Type image/jpeg -> "jpeg".
    resp_jpeg = MagicMock()
    resp_jpeg.content = b"not-an-image"
    resp_jpeg.headers = {"content-type": "image/jpeg"}
    resp_jpeg.raise_for_status = MagicMock()
    with patch("requests.get", return_value=resp_jpeg):
        result_jpeg = download_image_from_url("https://x/a")
    assert result_jpeg == (b"not-an-image", "jpeg")

    # 3) PIL fails, Content-Type image/svg+xml -> unsupported -> None.
    resp_svg = MagicMock()
    resp_svg.content = b"not-an-image"
    resp_svg.headers = {"content-type": "image/svg+xml"}
    resp_svg.raise_for_status = MagicMock()
    with patch("requests.get", return_value=resp_svg):
        assert download_image_from_url("https://x/a") is None

    # 4) requests.get raising -> never raises, returns None.
    with patch("requests.get", side_effect=RuntimeError("boom")):
        assert download_image_from_url("https://x/a") is None


def test_on_chat_model_start_real_image_uuids_attribute(client_with_mocked_transport):
    """Driving a chat model with a multimodal base64 image yields a captured
    span carrying llm.input.image_count==1 and a non-empty image_uuids list,
    and triggers an image upload; a no-image run leaves both keys absent."""
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage

    import noveum_trace
    from noveum_trace.integrations.langchain import NoveumTraceCallbackHandler

    client = client_with_mocked_transport

    # --- with image ---
    handler = NoveumTraceCallbackHandler()
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    img_msg = HumanMessage(
        content=[
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": _png_data_uri()}},
        ]
    )
    model.invoke([img_msg], config={"callbacks": [handler]})
    noveum_trace.flush()

    span = find_span(
        client, predicate=lambda s: "llm.input.image_count" in (s.attributes or {})
    )
    a = span.attributes
    assert a["llm.input.image_count"] == 1
    uuids = a["llm.input.image_uuids"]
    assert isinstance(uuids, list) and len(uuids) == 1
    assert isinstance(uuids[0], str) and uuids[0]
    assert a["llm.operation"] == "chat"
    status = getattr(span.status, "value", span.status)
    assert status == "ok"
    assert client.transport.export_image.call_count == 1

    # --- without image: a fresh run produces a span lacking both keys ---
    client.transport.export_trace.reset_mock()
    client.transport.export_image.reset_mock()
    handler2 = NoveumTraceCallbackHandler()
    model2 = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    text_msg = HumanMessage(content="plain text only")
    model2.invoke([text_msg], config={"callbacks": [handler2]})
    noveum_trace.flush()

    spans = get_exported_spans(client)
    assert spans, "expected at least one exported span for the no-image run"
    for s in spans:
        sa = s.attributes or {}
        assert "llm.input.image_count" not in sa
        assert "llm.input.image_uuids" not in sa
    assert client.transport.export_image.call_count == 0
