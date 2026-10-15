"""End-to-end local verification of Stage 4 in creaky-apex-cleanlab-blend.ipynb.

Verifies:
1. All 4 checkpoints load with strict=True:
   - 3x N4 ConvNeXt-Small (256x256, raw_v1)
   - 1x C384 CoAtNet-1 (384x384, cache_v1_384)
2. Forward pass runs cleanly on sample_dicom at both 256x256 and 384x384 without shape or position-bias errors.
3. Adaptive C384 + N4 weighting and 3-way rank fusion produce valid, finite probabilities in [0, 1].
"""

import json
import os
import tempfile
import numpy as np
import pandas as pd

APEX_NB = "notebooks/apex_cleanlab_blend/creaky-apex-cleanlab-blend.ipynb"


def main():
    with open(APEX_NB, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cell11_src = "".join(nb["cells"][11]["source"])
    # Extract everything up to the top-level execution block (`import shutil as _sh`)
    fn_defs = cell11_src.split("import shutil as _sh")[0]

    ns = {"__name__": "__test__"}
    exec(compile(fn_defs, "<stage4_cell11>", "exec"), ns)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_csv = os.path.join(tmpdir, "our_submission.csv")
        sub_df = ns["run_submission_inference"](out_csv)

        assert os.path.exists(out_csv), "our_submission.csv was not written"
        assert sub_df.shape[1] == 13, f"Expected 13 columns, got {sub_df.shape[1]}"
        assert len(sub_df) > 0, "Submission DataFrame is empty"
        vals = sub_df.iloc[:, 1:].values
        assert np.isfinite(vals).all(), "Predictions contain NaN/Inf"
        assert ((vals >= 0.001) & (vals <= 0.999)).all(), "Predictions outside [0.001, 0.999]"
        print("\n[PASS] Stage 4 end-to-end local inference verified across all 4 checkpoints (3x N4 256px + 1x C384 384px)!")


if __name__ == "__main__":
    main()
