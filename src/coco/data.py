"""
CaptionCOCO Dataset
===================
Image-caption multimodal dataset built on COCO 2017.

Each sample produces:
  - ``image``:   RGB, ImageNet-normalised             [3, 224, 224]
  - ``caption``: Tokenised caption (+ optional nuisance prefix)  [max_len]

Noise mechanisms
----------------
Image (strong modality)
  * Gaussian pixel noise  (``image_noise_std``)
  * Gaussian blur         (``image_blur_sigma``)  -- fixed sigma, sweep param
  * JPEG compression      (``image_jpeg_quality``) -- sweep param
  * Random cutout         (``image_cutout_fraction``) -- sweep param
  * Colour jitter + stochastic blur (``augment=True``, training only)

Text (weak modality)
  * Word dropout      (``text_dropout_prob``)
  * Word swap         (``text_swap_prob``)
  * Random insertion  (``text_insert_prob``)
  * Character noise   (``text_char_noise_prob``)  -- delete / sub / insert / transpose
  * Synonym replace   (``text_synonym_prob``)     -- WordNet/NLTK, silently skipped if unavailable

Nuisance injection (variance trap) — three modes via ``nuisance_mode``
  * **text_only** (original): inject image-stat tokens into caption text only.
  * **style**: apply visual style transforms to the image AND prepend coherent
    style descriptions to the caption.  Both modalities carry the nuisance.
  * **scene**: inject scene-context descriptions (non-dominant COCO objects)
    into the caption.  The image already contains these objects visually.

Labels: dominant-object category (largest bounding-box area), 80 COCO classes.
"""

import io
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

IMAGE_SIZE  = 224
NUM_CLASSES = 80

DEFAULT_COCO_ROOT = os.environ.get("COCO_ROOT", "/path/to/coco2017")

# -- Tokenisation ----------------------------------------------------------

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SEP_TOKEN = "<sep>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN]
PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
SEP_IDX = 4

MIN_WORD_FREQ  = 3
MAX_VOCAB_SIZE = 10_000
DEFAULT_MAX_CAPTION_LEN = 64

# Categorical bins for nuisance features (added to vocab automatically)
NUISANCE_TOKENS = [
    # brightness (6 bins)
    "very_dark", "dark", "dim", "moderate_brightness", "bright", "very_bright",
    # colour temperature (3 bins)
    "cool_tones", "neutral_tones", "warm_tones",
    # contrast (4 bins)
    "flat_contrast", "low_contrast", "medium_contrast", "high_contrast",
    # saturation (4 bins)
    "desaturated", "muted_color", "vivid_color", "very_vivid",
    # edge density (4 bins)
    "smooth_texture", "some_edges", "detailed_texture", "very_detailed",
    # spatial layout (3 bins)
    "top_heavy", "vertically_balanced", "bottom_heavy",
]

NUISANCE_MODES = ("text_only", "style", "scene", "caption_swap")

_EXCLUDED_FROM_INSERTION = set(SPECIAL_TOKENS) | set(NUISANCE_TOKENS)


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Lowercase whitespace + punctuation split."""
    return re.findall(r"[\w]+|[^\s\w]", text.lower().strip())


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """Word-level vocabulary with caching to disk."""

    def __init__(self) -> None:
        self.word2idx: Dict[str, int] = {}
        self.idx2word: List[str] = []
        for tok in SPECIAL_TOKENS:
            self._add(tok)
        for tok in NUISANCE_TOKENS:
            self._add(tok)

    def _add(self, word: str) -> int:
        if word not in self.word2idx:
            idx = len(self.idx2word)
            self.word2idx[word] = idx
            self.idx2word.append(word)
            return idx
        return self.word2idx[word]

    def __len__(self) -> int:
        return len(self.idx2word)

    # -- build / save / load -----------------------------------------------

    @classmethod
    def build_from_captions(
        cls,
        captions: List[str],
        min_freq: int = MIN_WORD_FREQ,
        max_size: int = MAX_VOCAB_SIZE,
    ) -> "Vocabulary":
        vocab = cls()
        counter: Counter = Counter()
        for cap in captions:
            counter.update(tokenize(cap))
        for word, freq in counter.most_common():
            if freq < min_freq or len(vocab) >= max_size:
                break
            vocab._add(word)
        return vocab

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"word2idx": self.word2idx, "idx2word": self.idx2word}, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        vocab = object.__new__(cls)
        with open(path) as f:
            data = json.load(f)
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = data["idx2word"]
        return vocab

    def ensure_tokens(self, tokens: List[str]) -> None:
        """Add *tokens* to the vocabulary if not already present."""
        for tok in tokens:
            self._add(tok)

    # -- encode / decode ---------------------------------------------------

    def encode(self, text: str, max_len: int) -> Tuple[List[int], int]:
        """Tokenise *text* and return ``(padded_ids, actual_length)``."""
        tokens = [BOS_TOKEN] + tokenize(text) + [EOS_TOKEN]
        ids = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        if len(ids) > max_len:
            ids = ids[: max_len - 1] + [EOS_IDX]
        length = len(ids)
        ids += [PAD_IDX] * (max_len - length)
        return ids, length

    def decode(self, ids, skip_special: bool = True) -> str:
        """Convert token IDs back to text, stopping at first <pad>."""
        words = []
        for idx in ids:
            if hasattr(idx, "item"):
                idx = idx.item()
            if idx == PAD_IDX:
                break
            word = self.idx2word[idx] if 0 <= idx < len(self.idx2word) else UNK_TOKEN
            if skip_special and word in (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN):
                continue
            words.append(word)
        return " ".join(words)


# ---------------------------------------------------------------------------
# Annotation loaders
# ---------------------------------------------------------------------------

def _load_coco_annotations(
    ann_file: str,
) -> Tuple[List[dict], Dict[int, int], List[str]]:
    """
    Load instance annotations and assign each image a single label:
    the category of its dominant object (largest bounding-box area).
    Returns ``(image_entries, cat_id_to_idx, cat_names)``.
    """
    with open(ann_file) as f:
        data = json.load(f)

    categories = sorted(data["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(categories)}
    cat_names = [cat["name"] for cat in categories]

    img_dominant: Dict[int, Tuple[int, float]] = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        area = ann["area"]
        if img_id not in img_dominant or area > img_dominant[img_id][1]:
            img_dominant[img_id] = (ann["category_id"], area)

    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    image_entries = []
    for img_id, (cat_id, _) in img_dominant.items():
        if img_id not in id_to_filename:
            continue
        image_entries.append({
            "file_name": id_to_filename[img_id],
            "label": cat_id_to_idx[cat_id],
            "image_id": img_id,
        })
    image_entries.sort(key=lambda e: e["image_id"])
    return image_entries, cat_id_to_idx, cat_names


def _load_coco_captions(ann_file: str) -> Dict[int, List[str]]:
    """Load caption annotations -> ``{image_id: [caption, ...]}``."""
    with open(ann_file) as f:
        data = json.load(f)
    captions: Dict[int, List[str]] = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        captions.setdefault(img_id, []).append(ann["caption"].strip())
    return captions


# ---------------------------------------------------------------------------
# Nuisance features
# ---------------------------------------------------------------------------

def _bin(value: float, thresholds: List[float], labels: List[str]) -> str:
    for thr, lbl in zip(thresholds, labels):
        if value < thr:
            return lbl
    return labels[-1]


def compute_nuisance_features(img: torch.Tensor) -> Dict[str, str]:
    """
    Compute categorical image statistics from a ``[3, H, W]`` tensor in [0, 1].
    Returns a dict mapping feature name -> categorical token.
    """
    r, g, b = img[0], img[1], img[2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    brightness = gray.mean().item()
    contrast   = gray.std().item()
    color_temp = (r.mean() - b.mean()).item()

    max_c = img.max(dim=0)[0]
    min_c = img.min(dim=0)[0]
    saturation = ((max_c - min_c) / (max_c + 1e-8)).mean().item()

    dx = (gray[:, 1:] - gray[:, :-1]).abs().mean().item()
    dy = (gray[1:, :] - gray[:-1, :]).abs().mean().item()
    edge_density = (dx + dy) / 2

    h = gray.shape[0]
    vert_balance = (gray[: h // 2].mean() - gray[h // 2 :].mean()).item()

    return {
        "brightness": _bin(
            brightness, [0.15, 0.25, 0.40, 0.55, 0.70],
            ["very_dark", "dark", "dim", "moderate_brightness",
             "bright", "very_bright"],
        ),
        "color_temperature": _bin(
            color_temp, [-0.04, 0.04],
            ["cool_tones", "neutral_tones", "warm_tones"],
        ),
        "contrast": _bin(
            contrast, [0.10, 0.18, 0.27],
            ["flat_contrast", "low_contrast", "medium_contrast",
             "high_contrast"],
        ),
        "saturation": _bin(
            saturation, [0.12, 0.25, 0.45],
            ["desaturated", "muted_color", "vivid_color", "very_vivid"],
        ),
        "edge_density": _bin(
            edge_density, [0.02, 0.04, 0.07],
            ["smooth_texture", "some_edges", "detailed_texture",
             "very_detailed"],
        ),
        "spatial_layout": _bin(
            vert_balance, [-0.04, 0.04],
            ["bottom_heavy", "vertically_balanced", "top_heavy"],
        ),
    }


_NUISANCE_ORDER = [
    "brightness", "color_temperature", "contrast",
    "saturation", "edge_density", "spatial_layout",
]


def format_nuisance_prefix(features: Dict[str, str], strength: float) -> str:
    """
    ``strength`` in [0, 1] controls how many features are injected:
    0.0 -> nothing,  ~0.17 -> 1 feature,  ~0.5 -> 3 features,  1.0 -> all 6.
    """
    if strength <= 0:
        return ""
    n = max(1, round(strength * len(_NUISANCE_ORDER)))
    return " ".join(features[name] for name in _NUISANCE_ORDER[:n])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CaptionCOCO(Dataset):
    """
    Image-Caption COCO 2017 dataset for multimodal learning experiments.

    Parameters
    ----------
    split : {'train', 'val'}
    coco_root : Root directory with train2017/, val2017/, annotations/.
    nuisance_strength : Fraction of nuisance features injected [0-1].
    nuisance_mode : How nuisance is injected (see module docstring).
        'text_only'  — original: image-stat tokens in caption only.
        'style'      — visual style transforms + text descriptions.
        'scene'      — scene-context descriptions from COCO annotations.

    Image noise (applied after augmentations; useful as sweep parameters)
    ----------
    image_noise_std : Additive Gaussian pixel noise std.
    image_blur_sigma : Fixed Gaussian blur sigma (0 = off).
    image_jpeg_quality : JPEG compression quality (100 = off, lower = more artefacts).
    image_cutout_fraction : Fraction of image area zeroed out (0 = off).

    Text noise (applied to caption before tokenisation)
    ----------
    text_dropout_prob : Per-word deletion probability.
    text_swap_prob : Per-adjacent-pair transposition probability.
    text_insert_prob : Per-word random insertion probability.
    text_char_noise_prob : Per-character noise probability (delete/sub/insert/transpose).
    text_synonym_prob : Per-word synonym replacement probability (needs NLTK WordNet).

    General
    -------
    max_caption_len : Fixed token-sequence length (padded / truncated).
    augment : Stochastic colour jitter + blur (training only, not probe).
    seed_augmentation : Seed RNG per sample index for reproducibility.
    vocab : Pre-built Vocabulary shared across splits.
    """

    def __init__(
        self,
        split: Literal["train", "val"] = "train",
        coco_root: str = DEFAULT_COCO_ROOT,
        nuisance_strength: float = 0.0,
        nuisance_mode: str = "text_only",
        style_jitter: float = 0.0,
        image_style_strength: float = 0.0,
        # -- image noise --
        image_noise_std: float = 0.0,
        image_blur_sigma: float = 0.0,
        image_jpeg_quality: int = 100,
        image_cutout_fraction: float = 0.0,
        # -- text noise --
        text_dropout_prob: float = 0.0,
        text_swap_prob: float = 0.0,
        text_insert_prob: float = 0.0,
        text_char_noise_prob: float = 0.0,
        text_synonym_prob: float = 0.0,
        # -- general --
        max_caption_len: int = DEFAULT_MAX_CAPTION_LEN,
        augment: bool = True,
        seed_augmentation: bool = False,
        vocab: Optional[Vocabulary] = None,
    ) -> None:
        super().__init__()
        if nuisance_mode not in NUISANCE_MODES:
            raise ValueError(
                f"Unknown nuisance_mode={nuisance_mode!r}. "
                f"Choose from {NUISANCE_MODES}."
            )
        self.coco_root            = Path(coco_root)
        self.split                = split
        self.nuisance_strength    = nuisance_strength
        self.nuisance_mode        = nuisance_mode
        self.style_jitter         = style_jitter
        self.image_style_strength = image_style_strength
        self.image_noise_std      = image_noise_std
        self.image_blur_sigma     = image_blur_sigma
        self.image_jpeg_quality   = image_jpeg_quality
        self.image_cutout_fraction = image_cutout_fraction
        self.text_dropout_prob    = text_dropout_prob
        self.text_swap_prob       = text_swap_prob
        self.text_insert_prob     = text_insert_prob
        self.text_char_noise_prob = text_char_noise_prob
        self.text_synonym_prob    = text_synonym_prob
        self.max_caption_len      = max_caption_len
        self._augment             = augment
        self._seed_aug            = seed_augmentation

        # -- Paths ---------------------------------------------------------
        if split == "train":
            self.img_dir   = self.coco_root / "train2017"
            instances_file = self.coco_root / "annotations" / "instances_train2017.json"
            captions_file  = self.coco_root / "annotations" / "captions_train2017.json"
        elif split == "val":
            self.img_dir   = self.coco_root / "val2017"
            instances_file = self.coco_root / "annotations" / "instances_val2017.json"
            captions_file  = self.coco_root / "annotations" / "captions_val2017.json"
        else:
            raise ValueError(f"Unknown split: {split!r}. Choose 'train' or 'val'.")

        for p, desc in [
            (instances_file, "Instance annotations"),
            (captions_file,  "Caption annotations"),
            (self.img_dir,   "Image directory"),
        ]:
            if not p.exists():
                raise FileNotFoundError(f"{desc} not found: {p}")

        # -- Load annotations ---------------------------------------------
        print(f"Loading COCO 2017 (split={split})...")
        self.image_entries, self.cat_id_to_idx, self.cat_names = \
            _load_coco_annotations(str(instances_file))
        self.captions_by_image = _load_coco_captions(str(captions_file))
        print(f"  {len(self.image_entries)} images, "
              f"{len(self.cat_names)} categories, "
              f"{sum(len(v) for v in self.captions_by_image.values())} captions.")

        # -- Vocabulary ----------------------------------------------------
        vocab_path = self.coco_root / "vocab_caption_coco.json"
        if vocab is not None:
            self.vocab = vocab
        elif vocab_path.exists():
            self.vocab = Vocabulary.load(str(vocab_path))
            print(f"  Vocabulary loaded from cache ({len(self.vocab)} tokens).")
        else:
            all_caps = [c for caps in self.captions_by_image.values() for c in caps]
            self.vocab = Vocabulary.build_from_captions(all_caps)
            try:
                self.vocab.save(str(vocab_path))
                print(f"  Vocabulary built and cached ({len(self.vocab)} tokens).")
            except OSError:
                print(f"  Vocabulary built ({len(self.vocab)} tokens, "
                      f"cache write failed).")

        # -- Mode-specific setup -------------------------------------------
        if self.nuisance_mode == "style" or self.image_style_strength > 0:
            from src.coco.nuisance_modes import STYLE_VOCAB_TOKENS
            self.vocab.ensure_tokens(STYLE_VOCAB_TOKENS)

        if self.nuisance_mode == "scene":
            from src.coco.nuisance_modes import load_scene_objects, SCENE_VOCAB_TOKENS
            self.vocab.ensure_tokens(SCENE_VOCAB_TOKENS)
            self.scene_objects, self.image_dims = load_scene_objects(
                str(instances_file), self.cat_names, self.cat_id_to_idx,
            )
            n_with_ctx = sum(
                1 for img_id in self.scene_objects
                if len(self.scene_objects[img_id]) > 1
            )
            print(f"  Scene context: {n_with_ctx} images have "
                  f"non-dominant objects.")

        if self.nuisance_mode == "caption_swap":
            # Build label -> list of dataset indices for within-category swapping
            from collections import defaultdict
            self._label_to_indices: Dict[int, List[int]] = defaultdict(list)
            for i, e in enumerate(self.image_entries):
                self._label_to_indices[e["label"]].append(i)
            self._label_to_indices = dict(self._label_to_indices)
            avg_per_cat = np.mean([len(v) for v in self._label_to_indices.values()])
            print(f"  Caption-swap mode: {len(self._label_to_indices)} categories, "
                  f"avg {avg_per_cat:.0f} images/category.")

        # Words eligible for random insertion (no special/nuisance, alpha only)
        self._insert_words: List[str] = [
            w for w in self.vocab.idx2word
            if w not in _EXCLUDED_FROM_INSERTION and w.isalpha() and len(w) > 2
        ]

        # -- Image transforms ----------------------------------------------
        self._resize_crop = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
        self._normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        # Stronger stochastic blur for augmentation (sigma up to 4.0)
        self._color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1,
        )
        self._augment_blur = transforms.GaussianBlur(
            kernel_size=25, sigma=(0.3, 4.0),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_idx(self) -> int:
        return PAD_IDX

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_entries)

    def __getitem__(self, idx: int) -> Dict:
        rng = (np.random.RandomState(idx + 271828)
               if self._seed_aug else np.random.RandomState())

        entry = self.image_entries[idx]
        img = Image.open(self.img_dir / entry["file_name"]).convert("RGB")
        label = entry["label"]

        # -- Image processing ---------------------------------------------
        img_tensor = self._resize_crop(img)  # [3, 224, 224], float32 in [0, 1]

        # --- Nuisance: compute text-only stats from clean pixels ----------
        nuisance_text = ""
        if self.nuisance_mode == "text_only":
            nuisance = compute_nuisance_features(img_tensor)
            nuisance_text = format_nuisance_prefix(
                nuisance, self.nuisance_strength,
            )

        # Stochastic augmentations (training only)
        if self._augment:
            if rng.random() < 0.5:
                img_tensor = TF.hflip(img_tensor)
            if rng.random() < 0.8:
                img_tensor = self._color_jitter(img_tensor)
            if rng.random() < 0.5:
                img_tensor = self._augment_blur(img_tensor)

        # --- Nuisance: style transforms (after augmentation) --------------
        if self.nuisance_mode == "style" and self.nuisance_strength > 0:
            from src.coco.nuisance_modes import apply_style_nuisance
            img_tensor, nuisance_text = apply_style_nuisance(
                img_tensor, self.nuisance_strength, rng,
                jitter=self.style_jitter, jitter_seed=idx,
            )

        # --- Image-only style noise (capswap v2: weak modality noise) -----
        if self.image_style_strength > 0 and self.nuisance_mode != "style":
            from src.coco.nuisance_modes import apply_style_nuisance
            img_tensor, _ = apply_style_nuisance(
                img_tensor, self.image_style_strength, rng,
            )
            # Gaussian blur scaled by style strength (sigma up to 4.0)
            blur_sigma = self.image_style_strength * 4.0
            blur_ks = 2 * int(3 * blur_sigma) + 1
            blur_ks = max(blur_ks, 3)
            img_tensor = TF.gaussian_blur(
                img_tensor, kernel_size=blur_ks, sigma=blur_sigma,
            )

        # Fixed Gaussian blur (sweep param)
        if self.image_blur_sigma > 0:
            ks = 2 * int(3 * self.image_blur_sigma) + 1
            ks = max(ks, 3)
            img_tensor = TF.gaussian_blur(
                img_tensor, kernel_size=ks, sigma=self.image_blur_sigma,
            )

        # JPEG compression (sweep param)
        if self.image_jpeg_quality < 100:
            buf = io.BytesIO()
            TF.to_pil_image(img_tensor).save(
                buf, format="JPEG", quality=self.image_jpeg_quality,
            )
            buf.seek(0)
            img_tensor = TF.to_tensor(Image.open(buf).convert("RGB"))

        # Random cutout (sweep param) — zeroes a square patch
        if self.image_cutout_fraction > 0:
            _, H, W = img_tensor.shape
            side = int((H * W * self.image_cutout_fraction) ** 0.5)
            x = int(rng.randint(0, max(1, W - side + 1)))
            y = int(rng.randint(0, max(1, H - side + 1)))
            img_tensor[:, y: y + side, x: x + side] = 0.0

        # Additive Gaussian noise (sweep param)
        if self.image_noise_std > 0:
            noise = torch.tensor(
                rng.normal(0, self.image_noise_std, size=img_tensor.shape),
                dtype=torch.float32,
            )
            img_tensor = (img_tensor + noise).clamp(0.0, 1.0)

        image = self._normalize(img_tensor)

        # -- Caption processing --------------------------------------------
        image_id = entry["image_id"]
        captions = self.captions_by_image.get(image_id, ["a photo"])
        raw_caption = captions[rng.randint(0, len(captions))]

        # --- Nuisance: within-category caption swap -----------------------
        if self.nuisance_mode == "caption_swap" and self.style_jitter > 0:
            if rng.random() < self.style_jitter:
                # Pick a different image from the same category
                same_cat = self._label_to_indices[label]
                donor_idx = same_cat[rng.randint(0, len(same_cat))]
                donor_id = self.image_entries[donor_idx]["image_id"]
                donor_caps = self.captions_by_image.get(donor_id, ["a photo"])
                raw_caption = donor_caps[rng.randint(0, len(donor_caps))]

        # --- Nuisance: scene context description --------------------------
        if self.nuisance_mode == "scene" and self.nuisance_strength > 0:
            nuisance_text = self._get_scene_context(entry)

        # --- Inject nuisance text into caption ----------------------------
        if self.nuisance_mode == "text_only" and nuisance_text:
            # Random-position injection with <sep>
            words = raw_caption.split()
            insert_pos = int(rng.randint(0, len(words) + 1))
            nuis_toks = nuisance_text.split()
            injected = (words[:insert_pos] + nuis_toks
                        + [SEP_TOKEN] + words[insert_pos:])
            full_text = " ".join(injected)
        elif self.nuisance_mode == "style" and nuisance_text:
            if self.style_jitter > 0:
                # Interleave style words into caption at random positions
                cap_words = raw_caption.split()
                nuis_words = nuisance_text.split()
                for w in nuis_words:
                    pos = int(rng.randint(0, len(cap_words) + 1))
                    cap_words.insert(pos, w)
                full_text = " ".join(cap_words)
            else:
                # jitter=0: prepend as coherent block (backward compat)
                full_text = f"{nuisance_text} {raw_caption}"
        elif self.nuisance_mode == "scene" and nuisance_text:
            # Append scene context after <sep>
            full_text = f"{raw_caption} {SEP_TOKEN} {nuisance_text}"
        else:
            full_text = raw_caption

        # Apply text noise to the assembled caption (including nuisance)
        full_text = self._apply_text_noise(full_text, rng)

        token_ids, length = self.vocab.encode(full_text, self.max_caption_len)

        return {
            "image":          image,
            "caption":        torch.tensor(token_ids, dtype=torch.long),
            "caption_length": length,
            "label":          label,
        }

    def _get_scene_context(self, entry: dict) -> str:
        """Build scene-context description for an image (scene mode only)."""
        from src.coco.nuisance_modes import format_scene_context
        image_id = entry["image_id"]
        objects = self.scene_objects.get(image_id, [])
        if not objects:
            return ""
        w, h = self.image_dims.get(image_id, (640, 480))
        return format_scene_context(
            objects, w, h, entry["label"], self.cat_names,
            self.nuisance_strength,
        )

    # ------------------------------------------------------------------
    # Text noise helpers
    # ------------------------------------------------------------------

    def _apply_text_noise(
        self, text: str, rng: np.random.RandomState,
    ) -> str:
        """Apply all enabled text noise mechanisms in sequence."""
        words = text.split()
        if not words:
            return text

        # 1. Word dropout
        if self.text_dropout_prob > 0:
            kept = [w for w in words if rng.random() > self.text_dropout_prob]
            words = kept if kept else words[:1]

        # 2. Word swap (adjacent transposition)
        if self.text_swap_prob > 0:
            for i in range(len(words) - 1):
                if rng.random() < self.text_swap_prob:
                    words[i], words[i + 1] = words[i + 1], words[i]

        # 3. Random insertion from vocabulary
        if self.text_insert_prob > 0 and self._insert_words:
            new_words = []
            for w in words:
                if rng.random() < self.text_insert_prob:
                    insert = self._insert_words[
                        rng.randint(0, len(self._insert_words))
                    ]
                    new_words.append(insert)
                new_words.append(w)
            words = new_words

        # 4. Random word replacement (from vocabulary)
        if self.text_char_noise_prob > 0 and self._insert_words:
            words = [
                self._insert_words[rng.randint(0, len(self._insert_words))]
                if rng.random() < self.text_char_noise_prob else w
                for w in words
            ]

        # 5. Synonym replacement (NLTK WordNet; no-op if unavailable)
        if self.text_synonym_prob > 0:
            words = self._synonym_replace(words, rng)

        return " ".join(words)

    def _char_noise(
        self, word: str, rng: np.random.RandomState,
    ) -> str:
        """Per-character noise: delete / substitute / insert / transpose."""
        if len(word) <= 1:
            return word
        chars = list(word)
        result: List[str] = []
        i = 0
        while i < len(chars):
            if rng.random() < self.text_char_noise_prob:
                op = rng.randint(0, 4)
                if op == 0 and len(chars) > 1:             # delete
                    i += 1
                elif op == 1:                               # substitute
                    result.append(chr(rng.randint(ord("a"), ord("z") + 1)))
                    i += 1
                elif op == 2:                               # insert before
                    result.append(chr(rng.randint(ord("a"), ord("z") + 1)))
                    result.append(chars[i])
                    i += 1
                elif op == 3 and i + 1 < len(chars):       # transpose
                    result.append(chars[i + 1])
                    result.append(chars[i])
                    i += 2
                else:
                    result.append(chars[i])
                    i += 1
            else:
                result.append(chars[i])
                i += 1
        return "".join(result) if result else word

    def _synonym_replace(
        self, words: List[str], rng: np.random.RandomState,
    ) -> List[str]:
        """Replace words with WordNet synonyms. Silently skips if NLTK unavailable."""
        try:
            from nltk.corpus import wordnet
        except (ImportError, LookupError):
            return words

        result = []
        for word in words:
            if rng.random() < self.text_synonym_prob:
                synonyms = [
                    lemma.name()
                    for syn in wordnet.synsets(word)
                    for lemma in syn.lemmas()
                    if lemma.name().lower() != word.lower()
                    and "_" not in lemma.name()  # skip multi-word entries
                ]
                if synonyms:
                    result.append(synonyms[rng.randint(0, len(synonyms))])
                    continue
            result.append(word)
        return result
