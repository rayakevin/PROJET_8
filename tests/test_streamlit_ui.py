import py_compile
from pathlib import Path


def test_streamlit_app_is_syntax_valid() -> None:
    py_compile.compile(
        str(Path("ui") / "streamlit_app.py"),
        doraise=True,
    )
