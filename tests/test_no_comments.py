import ast
import io
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {".venv", ".git", "__pycache__"}


def sources() -> list[Path]:
    return [p for p in ROOT.rglob("*") if p.suffix in (".py", ".pyi") and not SKIP & set(p.parts)]


def test_no_comments() -> None:
    for path in sources():
        tokens = tokenize.generate_tokens(io.StringIO(path.read_text()).readline)
        assert not [tok for tok in tokens if tok.type == tokenize.COMMENT], path


def test_no_docstrings() -> None:
    for path in sources():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and node.body:
                first = node.body[0]
                docstring = isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
                assert not docstring, path
