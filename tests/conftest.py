import sys
from pathlib import Path

# Add the src directory to the path so tests can import the package
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
