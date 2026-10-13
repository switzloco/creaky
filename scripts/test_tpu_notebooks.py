import json
import ast
from pathlib import Path

def test_notebooks():
    for f in [1, 2, 3]:
        p = Path(f"notebooks/training-book-n4-tpu-f{f}/training-book-n4-tpu-f{f}.ipynb")
        nb = json.loads(p.read_text(encoding="utf-8"))
        code = "".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"] if c["cell_type"] == "code")
        tree = ast.parse(code)
        print(f"Fold {f} syntax OK: {len(tree.body)} top-level AST nodes parsed.")
        
        # Verify key TPU blocks exist
        assert "HAS_XLA" in code, f"Fold {f} missing HAS_XLA"
        assert "xm.optimizer_step" in code, f"Fold {f} missing xm.optimizer_step"
        assert f'"fold": {f}' in code, f"Fold {f} missing fold {f} configuration"
        assert f"best_model_fold_{f}.pt" in code, f"Fold {f} missing best_model_fold_{f}.pt"
        print(f"Fold {f} verified successfully!")

if __name__ == "__main__":
    test_notebooks()
