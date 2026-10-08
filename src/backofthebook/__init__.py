"""Back of the Book: cited answers and practice quizzes from your course materials."""

from pathlib import Path as _Path

__version__ = "0.1.0"

# Newest modification time of this package's source files when it was imported. Long-running
# hosts compare it with the files on disk to notice that a deploy changed the code.
SOURCE_STAMP = max(p.stat().st_mtime for p in _Path(__file__).parent.glob("*.py"))
