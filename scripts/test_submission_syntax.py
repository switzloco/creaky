import json
import ast
from pathlib import Path

def test_syntax():
    p = Path("notebooks/apex_cleanlab_blend/creaky-apex-cleanlab-blend.ipynb")
    nb = json.loads(p.read_text(encoding="utf-8"))
    for idx, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            src = c["source"]
            lines = src.splitlines(keepends=True) if isinstance(src, str) else src
            # Filter magic commands
            filtered = [l for l in lines if not l.strip().startswith("%") and not l.strip().startswith("!")]
            code = "".join(filtered)
            try:
                tree = ast.parse(code)
                print(f"Cell {idx} parsed OK: {len(tree.body)} AST nodes.")
            except SyntaxError as e:
                print(f"SyntaxError in Cell {idx}: {e}")

if __name__ == "__main__":
    test_syntax()
