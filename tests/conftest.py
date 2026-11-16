import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["KEYPULSE_DB"] = os.path.join(tempfile.mkdtemp(), "test.db")
