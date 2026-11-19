<!-- Draft, not submitted. Issues are disabled on owlgebra-ai/faiss: the open
upstream PR is facebookresearch/faiss#5917. Reproduced on an NVIDIA L4. -->

# o-means fails on an NVIDIA L4 at k=100k: Triton out of shared memory in the coarse assignment

On an NVIDIA L4, `Omeans.train` fails at `k=100000` with the default
`nlist`/`nprobe`: the Triton kernel of the coarse assignment needs more CUDA
shared memory than the GPU has.

```python
import torch
from faiss.contrib.o_means import Omeans

n, d, k = 2_000_000, 96, 100_000
X = torch.randn(n, d, device="cuda")
init = X[torch.randperm(n, device="cuda")[:k]].clone()
Omeans(d, k, nepochs=1).train(X, init)
```

```
  File ".../faiss/contrib/o_means/o_means.py", line 369, in train
    labels = two_level_assign(src, self.c16, self.cn, self.cells, self.ws,
  File ".../faiss/contrib/o_means/triton_two_level.py", line 301, in two_level_assign
    coarse[(triton.cdiv(R, coarse_rows),)](
  ...
triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 135168, Hardware limit: 101376. Reducing block sizes or `num_stages` may help.
```

It also fails with `nlist=632`, and works with `nlist=316`. `k=10000` and
`k=30000` work with the defaults.

Versions: owlgebra-ai/faiss at 9ecbf0c, torch 2.13.0+cu129, triton 3.7.1,
NVIDIA L4 (24 GB).

I did not try a GPU with a higher shared memory limit, so this may only affect
the smaller ones. Maybe the block size of the coarse kernel could depend on
the limit of the device?
