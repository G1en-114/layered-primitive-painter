import numpy as np
import pytest
from PIL import Image, ImageOps

from lpp.data.anime import DuplicateIndex, face_crop_box, screen_tags, split_for_id, template_captions
from lpp.data.splits import load_split_manifest


def test_single_person_screening():
    assert screen_tags({"1girl", "solo"}) is None
    assert screen_tags({"1girl", "1boy"}) == "not_single_person"
    assert screen_tags({"2girls"}) == "not_single_person"
    assert screen_tags({"1girl", "nude"}) == "excluded_tags"


def test_crop_preserves_points_and_stays_in_image():
    points = np.stack([np.linspace(110, 240, 28), np.linspace(120, 230, 28)], 1)
    left, top, right, bottom = face_crop_box(points, (512, 512))
    assert 0 <= left < right <= 512 and 0 <= top < bottom <= 512
    assert right - left == bottom - top
    assert np.all(points >= [left, top]) and np.all(points < [right, bottom])


@pytest.mark.parametrize("points", [np.zeros((27, 2)), np.full((28, 2), np.nan), np.full((28, 2), -1)])
def test_invalid_landmarks_rejected(points):
    with pytest.raises(ValueError):
        face_crop_box(points, (512, 512))


def test_dedup_matches_mirrored_and_slightly_changed_images():
    rng = np.random.default_rng(123)
    pixels = rng.integers(15, 235, (256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    index = DuplicateIndex()
    assert index.add(image, 1) is None
    assert index.add(ImageOps.mirror(image), 2) == 1
    assert index.add(Image.fromarray(pixels + 1), 3) == 1
    assert index.add(Image.fromarray(rng.integers(0, 255, pixels.shape, dtype=np.uint8)), 4) is None


def test_caption_does_not_reuse_full_body_description():
    c = template_captions({"white_hair", "blue_eyes", "smile", "red_dress", "standing"})
    assert "白色头发" in c["long"] and "蓝色眼睛" in c["long"]
    assert "裙" not in c["long"] and "站" not in c["long"]


def test_split_is_stable_and_has_both_partitions():
    values = [split_for_id(200_000_000 + i) for i in range(1000)]
    assert values == [split_for_id(200_000_000 + i) for i in range(1000)]
    assert 20 < values.count("val") < 90


def test_explicit_split_manifest(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text('{"id": 200000001, "split": "train"}\n{"id": 200000002, "split": "val"}\n')
    assert load_split_manifest(path) == {200000001: "train", 200000002: "val"}


@pytest.mark.parametrize("text", [
    '{"id": 1, "split": "train"}\n{"id": 1, "split": "val"}\n',
    '{"id": 1, "split": "test"}\n',
    '',
])
def test_invalid_split_manifest_rejected(tmp_path, text):
    path = tmp_path / "manifest.jsonl"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_split_manifest(path)
