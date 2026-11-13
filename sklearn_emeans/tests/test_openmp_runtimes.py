"""Check that EMeans works next to the OpenMP runtime bundled in PyTorch.

On macOS, a wheel vendoring a libomp that exports weak symbols segfaults or
deadlocks next to the PyTorch wheel, depending on the import order, see
https://github.com/cakedev0/sklearn-contrib-emeans/issues/15.
"""

import subprocess
import sys

import pytest

pytest.importorskip("torch")

CODE = """
import numpy as np
{imports}

X = np.random.default_rng(0).standard_normal((200_000, 32), dtype=np.float32)

def emeans():
    EMeans(n_clusters=64, init="random", random_state=0).fit(X)

def pytorch():
    # A reduction large enough to run in the OpenMP threads of PyTorch.
    torch.ones(4_000_000).sum().item()

for step in [emeans, pytorch, emeans, pytorch]:
    step()
print("done")
"""


@pytest.mark.parametrize(
    "imports",
    [
        "import torch\nfrom sklearn_emeans import EMeans",
        "from sklearn_emeans import EMeans\nimport torch",
    ],
    ids=["torch-first", "emeans-first"],
)
def test_emeans_next_to_pytorch(imports):
    try:
        result = subprocess.run(
            [sys.executable, "-c", CODE.format(imports=imports)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("Deadlock: the fits next to PyTorch did not end in 60 s.")
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "done"
