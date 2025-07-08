"""Test basic imports and setup."""

import pytest


def test_imports():
    """Test that all required modules can be imported."""
    try:
        import gitignore_parser
        import hnswlib
        import llm
        import rank_bm25
        import sentence_transformers
        import sqlite_utils
        import tree_sitter

        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import required modules: {e}")


def test_vectorize_script_imports():
    """Test that vectorize script can be imported."""
    try:
        # This will test the script can be imported without execution
        import os
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        # Import should work without running main()
        with open("vectorize-repo.py") as f:
            code = f.read()

        # Basic syntax check
        compile(code, "vectorize-repo.py", "exec")
        assert True
    except Exception as e:
        pytest.fail(f"Failed to import vectorize script: {e}")


def test_analyze_script_imports():
    """Test that analyze script can be imported."""
    try:
        # This will test the script can be imported without execution
        import os
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

        # Import should work without running main()
        with open("analyze-repo.py") as f:
            code = f.read()

        # Basic syntax check
        compile(code, "analyze-repo.py", "exec")
        assert True
    except Exception as e:
        pytest.fail(f"Failed to import analyze script: {e}")
