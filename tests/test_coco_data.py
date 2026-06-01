"""
Tests for CaptionCOCO dataset.

Uses a small mock COCO dataset (images + instances + captions) generated
in a temp directory to verify shapes, dtypes, label ranges, nuisance
injection, text noise, image noise, and reproducibility.
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from src.coco.data import (
    CaptionCOCO,
    NUM_CLASSES,
    NUISANCE_TOKENS,
    SPECIAL_TOKENS,
    Vocabulary,
    compute_nuisance_features,
    format_nuisance_prefix,
)


# ---------------------------------------------------------------------------
# Fixtures: create a minimal mock COCO dataset
# ---------------------------------------------------------------------------

NUM_MOCK_IMAGES     = 20
NUM_MOCK_CATEGORIES = 5  # non-contiguous IDs, like real COCO

MOCK_CAPTIONS = [
    "a cat sitting on a wooden table",
    "a dog running in the park near a tree",
    "a red car parked on the street",
    "a person sitting in a comfortable chair",
    "a glass bottle on the kitchen counter",
]


def _make_mock_coco(root_dir: str, split: str = "train"):
    """
    Create a tiny mock COCO dataset with images, instance annotations,
    and caption annotations.
    """
    img_dir = os.path.join(root_dir, f"{split}2017")
    ann_dir = os.path.join(root_dir, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    # Non-contiguous category IDs (like real COCO)
    categories = [
        {"id": 1,  "name": "cat",    "supercategory": "animal"},
        {"id": 3,  "name": "dog",    "supercategory": "animal"},
        {"id": 18, "name": "car",    "supercategory": "vehicle"},
        {"id": 25, "name": "chair",  "supercategory": "furniture"},
        {"id": 44, "name": "bottle", "supercategory": "kitchen"},
    ]

    images = []
    instance_annotations = []
    caption_annotations = []
    inst_id = 1
    cap_id  = 1

    for img_idx in range(NUM_MOCK_IMAGES):
        fname = f"{img_idx:012d}.jpg"
        img = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        img.save(os.path.join(img_dir, fname))

        images.append({
            "id": img_idx + 1,
            "file_name": fname,
            "width": 256,
            "height": 256,
        })

        # 1-3 instance annotations per image
        n_anns = (img_idx % 3) + 1
        for j in range(n_anns):
            cat = categories[(img_idx + j) % NUM_MOCK_CATEGORIES]
            area = 100.0 * (j + 1)
            instance_annotations.append({
                "id": inst_id,
                "image_id": img_idx + 1,
                "category_id": cat["id"],
                "bbox": [10, 10, 50, 50],
                "area": area,
                "iscrowd": 0,
            })
            inst_id += 1

        # 5 captions per image (standard COCO)
        for j in range(5):
            caption_annotations.append({
                "id": cap_id,
                "image_id": img_idx + 1,
                "caption": MOCK_CAPTIONS[(img_idx + j) % len(MOCK_CAPTIONS)],
            })
            cap_id += 1

    # Write instance annotations
    with open(os.path.join(ann_dir, f"instances_{split}2017.json"), "w") as f:
        json.dump({
            "images": images,
            "annotations": instance_annotations,
            "categories": categories,
        }, f)

    # Write caption annotations
    with open(os.path.join(ann_dir, f"captions_{split}2017.json"), "w") as f:
        json.dump({
            "images": images,
            "annotations": caption_annotations,
        }, f)

    return root_dir


@pytest.fixture(scope="module")
def mock_coco_root():
    """Create a temporary mock COCO dataset for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _make_mock_coco(tmpdir, split="train")
        _make_mock_coco(tmpdir, split="val")
        yield tmpdir


@pytest.fixture(scope="module")
def train_dataset(mock_coco_root):
    return CaptionCOCO(
        split="train",
        coco_root=mock_coco_root,
        nuisance_strength=0.0,
        image_noise_std=0.0,
        text_dropout_prob=0.0,
        augment=False,
        seed_augmentation=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCaptionCOCO:

    def test_dataset_length(self, train_dataset):
        assert len(train_dataset) == NUM_MOCK_IMAGES

    def test_image_shape_dtype(self, train_dataset):
        sample = train_dataset[0]
        assert sample["image"].shape == (3, 224, 224)
        assert sample["image"].dtype == torch.float32

    def test_caption_shape_dtype(self, train_dataset):
        sample = train_dataset[0]
        assert sample["caption"].shape == (64,)
        assert sample["caption"].dtype == torch.long

    def test_caption_length_range(self, train_dataset):
        sample = train_dataset[0]
        length = sample["caption_length"]
        # at least BOS + one word + EOS = 3
        assert 3 <= length <= 64

    def test_label_range(self, train_dataset):
        for i in range(min(10, len(train_dataset))):
            label = train_dataset[i]["label"]
            assert 0 <= label < NUM_MOCK_CATEGORIES, \
                f"Label {label} out of range [0, {NUM_MOCK_CATEGORIES})"

    def test_image_is_normalized(self, train_dataset):
        """After ImageNet normalization, pixel values should go below 0."""
        sample = train_dataset[0]
        assert sample["image"].min() < 0, \
            "image should be ImageNet-normalized (some values < 0)"

    def test_vocab_has_special_and_nuisance_tokens(self, train_dataset):
        vocab = train_dataset.vocab
        for tok in SPECIAL_TOKENS:
            assert tok in vocab.word2idx, f"Missing special token: {tok}"
        for tok in NUISANCE_TOKENS:
            assert tok in vocab.word2idx, f"Missing nuisance token: {tok}"

    def test_vocab_size(self, train_dataset):
        n_base = len(SPECIAL_TOKENS) + len(NUISANCE_TOKENS)
        assert train_dataset.vocab_size > n_base, \
            "Vocab should contain caption words beyond special/nuisance tokens"

    def test_nuisance_injection_adds_tokens(self, mock_coco_root):
        ds_none = CaptionCOCO(
            split="train", coco_root=mock_coco_root,
            nuisance_strength=0.0,
            augment=False, seed_augmentation=True,
        )
        ds_full = CaptionCOCO(
            split="train", coco_root=mock_coco_root,
            nuisance_strength=1.0,
            augment=False, seed_augmentation=True,
        )
        len_none = ds_none[0]["caption_length"]
        len_full = ds_full[0]["caption_length"]
        assert len_full > len_none, \
            f"Nuisance injection should add tokens (got {len_full} vs {len_none})"

    def test_text_dropout_shortens_caption(self, mock_coco_root):
        ds_clean = CaptionCOCO(
            split="train", coco_root=mock_coco_root,
            text_dropout_prob=0.0,
            augment=False, seed_augmentation=True,
        )
        ds_noisy = CaptionCOCO(
            split="train", coco_root=mock_coco_root,
            text_dropout_prob=0.5,
            augment=False, seed_augmentation=True,
        )
        # Over many samples, dropout should shorten at least some captions
        shorter_count = 0
        for i in range(len(ds_clean)):
            if ds_noisy[i]["caption_length"] < ds_clean[i]["caption_length"]:
                shorter_count += 1
        assert shorter_count > 0, "text_dropout_prob=0.5 should shorten some captions"

    def test_image_noise_changes_image(self, mock_coco_root):
        ds_clean = CaptionCOCO(
            split="train", coco_root=mock_coco_root,
            image_noise_std=0.0,
            augment=False, seed_augmentation=True,
        )
        ds_noisy = CaptionCOCO(
            split="train", coco_root=mock_coco_root,
            image_noise_std=0.5,
            augment=False, seed_augmentation=True,
        )
        diff = (ds_clean[0]["image"] - ds_noisy[0]["image"]).abs().mean()
        assert diff > 0.01, f"image_noise_std=0.5 should change image, diff={diff:.4f}"

    def test_reproducibility(self, train_dataset):
        """Same index with seed_augmentation=True should produce identical outputs."""
        s1 = train_dataset[5]
        s2 = train_dataset[5]
        assert torch.allclose(s1["image"], s2["image"]), "image not reproducible"
        assert torch.equal(s1["caption"], s2["caption"]), "caption not reproducible"
        assert s1["label"] == s2["label"], "label not reproducible"

    def test_dominant_object_label(self, mock_coco_root):
        """Verify label corresponds to the largest-area object."""
        ds = CaptionCOCO(
            split="train", coco_root=mock_coco_root,
            augment=False, seed_augmentation=True,
        )
        # Image index 1 (image_id=2) has 2 annotations:
        #   j=0 (area=100, cat=(1+0)%5=dog, id=3)
        #   j=1 (area=200, cat=(1+1)%5=car, id=18)
        # Dominant = largest area = car (cat_id=18 -> contiguous idx 2)
        sample = ds[1]
        assert sample["label"] == 2, \
            f"Expected label 2 (car, largest area), got {sample['label']}"


# ---------------------------------------------------------------------------
# Nuisance feature unit tests
# ---------------------------------------------------------------------------

class TestNuisanceFeatures:

    def test_compute_returns_all_keys(self):
        img = torch.rand(3, 64, 64)
        features = compute_nuisance_features(img)
        expected_keys = {
            "brightness", "color_temperature", "contrast",
            "saturation", "edge_density", "spatial_layout",
        }
        assert set(features.keys()) == expected_keys

    def test_all_values_are_known_tokens(self):
        img = torch.rand(3, 64, 64)
        features = compute_nuisance_features(img)
        for key, val in features.items():
            assert val in NUISANCE_TOKENS, \
                f"Feature '{key}' returned unknown token '{val}'"

    def test_format_strength_zero(self):
        features = {"brightness": "bright", "color_temperature": "warm_tones",
                     "contrast": "high_contrast", "saturation": "vivid_color",
                     "edge_density": "detailed_texture", "spatial_layout": "balanced"}
        assert format_nuisance_prefix(features, 0.0) == ""

    def test_format_strength_scales_features(self):
        features = {"brightness": "bright", "color_temperature": "warm_tones",
                     "contrast": "high_contrast", "saturation": "vivid_color",
                     "edge_density": "detailed_texture",
                     "spatial_layout": "vertically_balanced"}
        low  = format_nuisance_prefix(features, 0.2)
        high = format_nuisance_prefix(features, 1.0)
        assert len(high.split()) > len(low.split()), \
            "Higher strength should inject more features"

    def test_dark_image_gets_dark_bin(self):
        img = torch.full((3, 64, 64), 0.05)
        features = compute_nuisance_features(img)
        assert features["brightness"] in ("very_dark", "dark")

    def test_bright_image_gets_bright_bin(self):
        img = torch.full((3, 64, 64), 0.85)
        features = compute_nuisance_features(img)
        assert features["brightness"] in ("bright", "very_bright")


# ---------------------------------------------------------------------------
# Vocabulary unit tests
# ---------------------------------------------------------------------------

class TestVocabulary:

    def test_special_tokens_at_expected_indices(self):
        vocab = Vocabulary()
        assert vocab.word2idx["<pad>"] == 0
        assert vocab.word2idx["<unk>"] == 1
        assert vocab.word2idx["<bos>"] == 2
        assert vocab.word2idx["<eos>"] == 3
        assert vocab.word2idx["<sep>"] == 4

    def test_build_from_captions(self):
        captions = ["a cat on a mat", "a dog on a log", "the cat sat"]
        vocab = Vocabulary.build_from_captions(captions, min_freq=1)
        assert "cat" in vocab.word2idx
        assert "dog" in vocab.word2idx

    def test_encode_pad_and_truncate(self):
        vocab = Vocabulary.build_from_captions(
            ["hello world test"], min_freq=1,
        )
        ids, length = vocab.encode("hello world", max_len=10)
        assert len(ids) == 10
        assert length == 4  # BOS + hello + world + EOS
        assert ids[length:] == [0] * (10 - length)  # padding

        ids2, length2 = vocab.encode("hello world test extra words", max_len=4)
        assert len(ids2) == 4
        assert ids2[-1] == 3  # EOS forced at end

    def test_save_and_load(self, tmp_path):
        vocab = Vocabulary.build_from_captions(["a cat", "a dog"], min_freq=1)
        path = str(tmp_path / "vocab.json")
        vocab.save(path)
        loaded = Vocabulary.load(path)
        assert loaded.word2idx == vocab.word2idx
        assert loaded.idx2word == vocab.idx2word
