"""
Script to fetch traces from Noveum Platform and save them to a file.

Fetches traces for a specific project and environment using the noveum_platform methods.
"""

from __future__ import annotations

import json
from pathlib import Path

from novaeval.noveum_platform import NoveumClient

# Configuration
API_KEY = "nv_7k58kPWeL0fJf5zDbazy2zRUBdP91Ywk"
PROJECT = "wealthink-research-automation-v0"
ENVIRONMENT = "testing"

# Generate output filename based on project and environment
# Replace special characters that might not be filesystem-friendly
safe_project = PROJECT.replace("/", "_").replace("\\", "_")
safe_env = ENVIRONMENT.replace("/", "_").replace("\\", "_")
OUTPUT_FILE = Path(f"{safe_project}_{safe_env}_traces.json")


def fetch_all_traces(client: NoveumClient) -> list[dict[str, object]]:
    """
    Fetch all traces for the configured project and environment.
    Handles pagination to retrieve all available traces.
    """
    all_traces = []
    from_ = 0
    size = 100  # Maximum allowed size per request
    has_more = True

    print(f"Fetching traces for project '{PROJECT}' in environment '{ENVIRONMENT}'...")

    while has_more:
        response = client.query_traces(
            project=PROJECT,
            environment=ENVIRONMENT,
            size=size,
            from_=from_,
            include_spans=False,
            sort="start_time:desc",
        )

        if not response.get("success"):
            raise RuntimeError(f"Failed to query traces: {response}")

        traces = response.get("traces")
        if traces is None:
            traces = response.get("data")
        if traces is None:
            raise RuntimeError("Query traces response did not contain trace data.")

        # Some responses wrap the traces inside data["traces"]
        if isinstance(traces, dict) and "traces" in traces:
            traces = traces["traces"]

        if not isinstance(traces, list):
            raise TypeError("Expected traces to be a list of trace dictionaries.")

        all_traces.extend(traces)
        print(f"  Fetched {len(traces)} traces (total: {len(all_traces)})")

        # Check pagination info
        pagination = response.get("pagination", {})
        has_more = pagination.get("has_more", False)
        from_ += len(traces)

        # If we got fewer traces than requested, we've reached the end
        if len(traces) < size:
            has_more = False

    print(f"\nTotal traces fetched: {len(all_traces)}")
    return all_traces


def write_json(path: Path, data: object) -> None:
    """
    Write data to a JSON file with UTF-8 encoding and pretty indentation.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2)
    print(f"Traces saved to: {path.absolute()}")


def main() -> None:
    """Main function to fetch and save traces."""
    # Initialize client with provided API key
    client = NoveumClient(api_key=API_KEY)

    # Fetch all traces
    traces = fetch_all_traces(client)

    # Save to file
    write_json(OUTPUT_FILE, traces)


if __name__ == "__main__":
    main()

