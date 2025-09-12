#!/usr/bin/env python3
"""
Script to combine spans from all trace files into a single dataset.
Each span will be merged with its parent trace data, with clashing keys prefixed.
"""

import json
import os
from pathlib import Path


def combine_spans_from_traces(traces_dir):
    """
    Combine all spans from trace files into a single list.
    Each span gets merged with its parent trace data.
    """
    combined_spans = []

    # Get all trace files
    trace_files = sorted([f for f in os.listdir(
        traces_dir) if f.startswith('trace') and f.endswith('.json')])

    print(f"Found {len(trace_files)} trace files: {trace_files}")

    for trace_file in trace_files:
        file_path = os.path.join(traces_dir, trace_file)
        print(f"Processing {trace_file}...")

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract trace data (the single object in traces array)
            if 'traces' in data and len(data['traces']) > 0:
                trace = data['traces'][0]

                # Process each span in this trace
                for span in trace.get('spans', []):
                    # Create a new object that combines span and trace data
                    combined_span = {}

                    # Add all span fields first
                    for key, value in span.items():
                        combined_span[key] = value

                    # Add trace fields with prefix to avoid clashes
                    for key, value in trace.items():
                        if key != 'spans':  # Skip the spans array itself
                            # Check if key already exists in span
                            if key in span:
                                # Prefix with 'trace_' to avoid clash
                                combined_span[f'trace_{key}'] = value
                            else:
                                # No clash, add as is
                                combined_span[key] = value

                    combined_spans.append(combined_span)

        except Exception as e:
            print(f"Error processing {trace_file}: {e}")
            continue

    return combined_spans


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    traces_dir = str(script_dir)

    print(f"Processing traces from: {traces_dir}")

    # Combine all spans
    combined_spans = combine_spans_from_traces(traces_dir)

    print(f"Combined {len(combined_spans)} spans total")

    # Save to dataset.json
    output_file = os.path.join(traces_dir, 'dataset.json')

    with open(output_file, 'w') as f:
        json.dump(combined_spans, f, indent=2)

    print(f"Saved combined spans to: {output_file}")

    # Print some statistics
    if combined_spans:
        print(f"\nSample of first span keys: {list(combined_spans[0].keys())}")
        print(f"Total spans: {len(combined_spans)}")

        # Count spans by type
        span_types = {}
        for span in combined_spans:
            span_name = span.get('name', 'unknown')
            span_types[span_name] = span_types.get(span_name, 0) + 1

        print(f"\nSpan types distribution:")
        for span_type, count in sorted(span_types.items()):
            print(f"  {span_type}: {count}")


if __name__ == "__main__":
    main()
