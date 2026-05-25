---
name: feedback_tikhonov_gram_terminology
description: Tikhonov resolution comment in test_utils.py uses "gram diagonal" for what is actually the inverse-gram diagonal; the numeric derivation is correct but the term is wrong
metadata:
  type: feedback
---

In `tests/src/test_utils.py::test_solve_tikhonov_resolution_underdetermined`, the comment explains:

> "each block has gram diagonal 1.1/0.21"

But 1.1/0.21 is the (G^T G + λI)^{-1} diagonal (the *inverse*-gram diagonal), not the gram diagonal. The gram diagonal itself is 1.1. The formula `1 - λ * (G^T G + λI)^{-1}_{jj}` still yields the correct value 10/21. Flag this as a NIT when reviewing similar derivation comments.

**Why:** Confusing gram with inverse-gram is a common notation slip in regularization derivations; future reviewers could be misled.

**How to apply:** When reviewing inline derivation comments for Tikhonov/regularization, check that "gram" refers to G^T G (or G^T G + λI), not its inverse.
