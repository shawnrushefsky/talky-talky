"""Database module for audiobook project management."""

from .connection import get_database, close_database, get_audiobook_dir, get_current_project_path
from .schema import initialize_schema

__all__ = [
    "get_database",
    "close_database",
    "get_audiobook_dir",
    "get_current_project_path",
    "initialize_schema",
]
