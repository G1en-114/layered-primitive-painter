import json

import numpy as np
import pytest

from lpp.data.sequences import SequenceDataset, load_shards
from lpp.text import CaptionStore, mirror_caption
from test_sequence import fake_decomposition


@pytest.mark.parametrize("original,expected", [
    ("左侧头发遮住左眼，脸朝右。", "右侧头发遮住右眼，脸朝左。"),
    ("右上角明亮，向左看，右耳戴耳环。", "左上角明亮，向右看，左耳戴耳环。"),
    ("左右对称，左、右眼清晰，右边有窗户。", "左右对称，左、右眼清晰，左边有窗户。"),
    ("Left profile, RIGHT earring, bright background.", "Right profile, LEFT earring, bright background."),
    ("黑发，微笑，浅色背景。", "黑发，微笑，浅色背景。"),
    (None, None),
])
def test_direction_words_and_paired_expressions(original, expected):
    assert mirror_caption(original) == expected
    assert mirror_caption(expected) == original


def test_shard_flip_metadata_changes_only_caption(tmp_path, proto):
    tokens, layer, rounds = fake_decomposition(3)
    (tmp_path / "config.json").write_text(json.dumps({"flip": True}))
    path = tmp_path / "decomp_000.npz"
    np.savez(path, tokens=tokens.numpy(), layer=layer.numpy(), round=rounds.numpy(),
             offsets=[0, len(tokens)], ids=[42])
    examples = load_shards([path])
    assert examples[0].flip and examples[0].image_id == 42
    assert np.array_equal(examples[0].tokens.numpy(), tokens.numpy())
    store = CaptionStore()
    store.vlm[42] = {"long": "脸朝左，右耳戴耳环。", "short": "脸朝左，右耳戴耳环。"}
    dataset = SequenceDataset(examples, proto, captions=store, p_text_drop=0.0, p_redraw=0, p_refine=0)
    assert dataset[0][1] == "脸朝右，左耳戴耳环。"
