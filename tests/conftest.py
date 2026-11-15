import math

import pytest
import torch

from lpp.protocol import load_protocol
from lpp.render import Primitives


@pytest.fixture(scope="session")
def proto():
    return load_protocol()


def random_prims(B: int, N: int, seed: int = 0, dtype=torch.float32, device="cpu") -> Primitives:
    g = torch.Generator().manual_seed(seed)

    def r(*shape):
        return torch.rand(*shape, generator=g, dtype=dtype)

    return Primitives(
        xy=(r(B, N, 2) * 256).to(device),
        sigma=torch.exp(math.log(2) + r(B, N, 2) * math.log(30)).to(device),
        theta=(r(B, N) * math.pi).to(device),
        beta=torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=dtype)[torch.randint(4, (B, N), generator=g)].to(device),
        rgb=r(B, N, 3).to(device),
        alpha=(0.2 + 0.79 * r(B, N)).to(device),
    )


@pytest.fixture
def prims():
    return random_prims(2, 40)
