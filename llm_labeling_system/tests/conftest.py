"""Test setup: point the tool at a throwaway data dir and disable live API
calls BEFORE the package (and its settings singleton) is imported."""

import os
import tempfile

os.environ["LABELING_DATA_DIR"] = tempfile.mkdtemp(prefix="labtest_")
os.environ["DEEPSEEK_API_KEY"] = ""  # deterministic: tests never hit the network
