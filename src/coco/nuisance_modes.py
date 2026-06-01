"""
Alternative nuisance injection modes for CaptionCOCO.
=====================================================

Three modes for injecting correlated nuisance between image and caption:

**text_only** (original):
    Compute image statistics, inject as categorical tokens into caption.
    Problem: only text is affected; acts as uncorrelated noise in the
    weak modality rather than true shared nuisance.

**style** (Option A):
    Apply random visual style transforms to the image AND prepend coherent
    style descriptions to the caption.  Both modalities carry the nuisance
    natively: the image encoder sees styled pixels, the text encoder reads
    about the style.  Controlled by ``nuisance_strength`` (number of style
    groups activated).

**scene** (Option B):
    Extract non-dominant objects from COCO instance annotations and inject
    scene-context descriptions into the caption.  The image already contains
    these objects visually; the text now describes them explicitly.
    Controlled by ``nuisance_strength`` (number of context objects).
"""

import json
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torchvision.transforms import functional as TF


# ═══════════════════════════════════════════════════════════════════════
# Option A: Style Transforms
# ═══════════════════════════════════════════════════════════════════════

# --- Individual style transform functions ---
# Each takes img [3,H,W] in [0,1], a random intensity t ∈ [0,1], and an
# RNG.  Returns (transformed_img, intensity_adjective).
# The intensity makes every sample's nuisance unique even within the same
# style category, and the adjective encodes it in text.

_INTENSITY_WORDS = {
    # Quantise continuous t into one of 5 buckets for the text description.
    # Boundaries are approximate — exact t is only in the pixels.
    0: "slightly",
    1: "somewhat",
    2: "moderately",
    3: "strongly",
    4: "very strongly",
}


def _intensity_word(t: float) -> str:
    bucket = min(int(t * 5), 4)
    return _INTENSITY_WORDS[bucket]


def _style_sepia(
    img: torch.Tensor, t: float, rng: np.random.RandomState,
) -> torch.Tensor:
    """Sepia tone colour filter, blended by *t*."""
    r, g, b = img[0], img[1], img[2]
    sr = (0.393 * r + 0.769 * g + 0.189 * b).clamp(0, 1)
    sg = (0.349 * r + 0.686 * g + 0.168 * b).clamp(0, 1)
    sb = (0.272 * r + 0.534 * g + 0.131 * b).clamp(0, 1)
    sepia = torch.stack([sr, sg, sb])
    return img * (1 - t) + sepia * t


def _style_cool_blue(
    img: torch.Tensor, t: float, rng: np.random.RandomState,
) -> torch.Tensor:
    """Cool blue colour shift scaled by *t*."""
    shift = torch.zeros_like(img)
    shift[2] = 0.20 * t
    shift[0] = -0.10 * t
    return (img + shift).clamp(0, 1)


def _style_warm_amber(
    img: torch.Tensor, t: float, rng: np.random.RandomState,
) -> torch.Tensor:
    """Warm amber / sunset colour shift scaled by *t*."""
    shift = torch.zeros_like(img)
    shift[0] = 0.15 * t
    shift[1] = 0.06 * t
    shift[2] = -0.12 * t
    return (img + shift).clamp(0, 1)


def _style_green_cast(
    img: torch.Tensor, t: float, rng: np.random.RandomState,
) -> torch.Tensor:
    """Green colour cast scaled by *t*."""
    shift = torch.zeros_like(img)
    shift[1] = 0.15 * t
    shift[0] = -0.08 * t
    shift[2] = -0.08 * t
    return (img + shift).clamp(0, 1)


def _style_high_contrast(
    img: torch.Tensor, t: float, rng: np.random.RandomState,
) -> torch.Tensor:
    """Contrast boost: factor = 1 + 2*t (up to 3x)."""
    mean = img.mean()
    factor = 1.0 + 2.0 * t
    return ((img - mean) * factor + mean).clamp(0, 1)


def _style_low_contrast(
    img: torch.Tensor, t: float, rng: np.random.RandomState,
) -> torch.Tensor:
    """Contrast reduction: factor = 1 - 0.8*t (down to 0.2x)."""
    mean = img.mean()
    factor = 1.0 - 0.8 * t
    return ((img - mean) * factor + mean).clamp(0, 1)


def _style_overexposed(
    img: torch.Tensor, t: float, rng: np.random.RandomState,
) -> torch.Tensor:
    """Overexpose: multiply by 1+t and add 0.15*t."""
    return (img * (1.0 + t) + 0.15 * t).clamp(0, 1)


def _style_underexposed(
    img: torch.Tensor, t: float, rng: np.random.RandomState,
) -> torch.Tensor:
    """Underexpose: multiply by 1-0.7*t."""
    return (img * (1.0 - 0.7 * t)).clamp(0, 1)


def _style_grain(
    img: torch.Tensor, t: float, rng: np.random.RandomState,
) -> torch.Tensor:
    """Additive grain with std = 0.25*t."""
    noise = torch.tensor(
        rng.normal(0, 0.25 * t, size=img.shape), dtype=torch.float32,
    )
    return (img + noise).clamp(0, 1)


def _style_soft_focus(
    img: torch.Tensor, t: float, rng: np.random.RandomState,
) -> torch.Tensor:
    """Gaussian blur with sigma = 1 + 5*t."""
    sigma = 1.0 + 5.0 * t
    ks = 2 * int(3 * sigma) + 1
    ks = max(ks, 3)
    return TF.gaussian_blur(img, kernel_size=ks, sigma=sigma)


def _style_vivid(
    img: torch.Tensor, t: float, rng: np.random.RandomState,
) -> torch.Tensor:
    """Saturation boost: factor = 1 + 2.5*t."""
    gray = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2])
    gray = gray.unsqueeze(0).expand_as(img)
    factor = 1.0 + 2.5 * t
    return (gray + (img - gray) * factor).clamp(0, 1)


def _style_muted(
    img: torch.Tensor, t: float, rng: np.random.RandomState,
) -> torch.Tensor:
    """Desaturate: factor = 1 - 0.9*t."""
    gray = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2])
    gray = gray.unsqueeze(0).expand_as(img)
    factor = 1.0 - 0.9 * t
    return (gray + (img - gray) * factor).clamp(0, 1)


def _style_vignette(
    img: torch.Tensor, t: float, rng: np.random.RandomState,
) -> torch.Tensor:
    """Vignette with darkening proportional to *t*."""
    _, H, W = img.shape
    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij",
    )
    dist_sq = X ** 2 + Y ** 2
    mask = (1.0 - t * (dist_sq / 2.0).clamp(0, 1)).unsqueeze(0)
    return (img * mask).clamp(0, 1)


def _style_center_spotlight(
    img: torch.Tensor, t: float, rng: np.random.RandomState,
) -> torch.Tensor:
    """Brighten centre, darken edges — inverse vignette scaled by *t*."""
    _, H, W = img.shape
    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij",
    )
    dist_sq = X ** 2 + Y ** 2
    boost = (1.0 + t * 0.5 * (1.0 - dist_sq.clamp(0, 1))).unsqueeze(0)
    return (img * boost).clamp(0, 1)


def _style_posterize(
    img: torch.Tensor, t: float, rng: np.random.RandomState,
) -> torch.Tensor:
    """Posterize: levels = 8 - 6*t (8 down to 2)."""
    levels = max(2, round(8 - 6 * t))
    return ((img * levels).floor() / (levels - 1)).clamp(0, 1)


def _style_solarize(
    img: torch.Tensor, t: float, rng: np.random.RandomState,
) -> torch.Tensor:
    """Solarize: invert pixels above threshold = 1 - 0.5*t."""
    threshold = 1.0 - 0.5 * t
    result = img.clone()
    mask = result > threshold
    result[mask] = 1.0 - result[mask]
    return result


# --- Style catalogue (grouped to avoid conflicting transforms) ---

STYLE_GROUPS: List[List[dict]] = [
    # Group 0: Colour cast
    [
        {"fn": _style_sepia, "desc": "sepia toned"},
        {"fn": _style_cool_blue, "desc": "cool blue tinted"},
        {"fn": _style_warm_amber, "desc": "warm amber tinted"},
        {"fn": _style_green_cast, "desc": "green tinted"},
    ],
    # Group 1: Exposure
    [
        {"fn": _style_overexposed, "desc": "overexposed and washed out"},
        {"fn": _style_underexposed, "desc": "dark and underexposed"},
    ],
    # Group 2: Contrast
    [
        {"fn": _style_high_contrast, "desc": "high contrast"},
        {"fn": _style_low_contrast, "desc": "low contrast and flat"},
    ],
    # Group 3: Texture / effect
    [
        {"fn": _style_grain, "desc": "grainy and noisy"},
        {"fn": _style_soft_focus, "desc": "soft focus and blurry"},
        {"fn": _style_posterize, "desc": "posterized"},
        {"fn": _style_solarize, "desc": "solarized"},
    ],
    # Group 4: Saturation
    [
        {"fn": _style_vivid, "desc": "vivid and oversaturated"},
        {"fn": _style_muted, "desc": "muted and desaturated"},
    ],
    # Group 5: Spatial
    [
        {"fn": _style_vignette, "desc": "vignetted with dark edges"},
        {"fn": _style_center_spotlight, "desc": "bright center spotlight"},
    ],
]

N_STYLE_GROUPS = len(STYLE_GROUPS)

# Tokens that may not appear often enough in COCO captions to survive
# min-freq filtering.  Added to vocab dynamically via ensure_tokens().
STYLE_VOCAB_TOKENS: List[str] = [
    # Colour cast
    "sepia", "toned", "tinted", "amber", "cool",
    # Exposure
    "overexposed", "underexposed", "washed",
    # Contrast
    "contrast", "flat",
    # Texture / effect
    "grainy", "noisy", "posterized", "solarized", "blurry", "focus",
    # Saturation
    "vivid", "oversaturated", "muted", "desaturated",
    # Spatial
    "vignetted", "edges", "spotlight",
    # Intensity adjectives
    "slightly", "somewhat", "moderately", "strongly",
]


def apply_style_nuisance(
    img: torch.Tensor,
    strength: float,
    rng: np.random.RandomState,
    jitter: float = 0.0,
    jitter_seed: int = 0,
) -> Tuple[torch.Tensor, str]:
    """
    Apply random visual styles to the image and return a text description.

    Each active style gets a **random continuous intensity** t ~ U(0.3, 1.0),
    making every sample's pixel-level nuisance unique.  The intensity is
    quantised into 5 adjective buckets for the text description (slightly /
    somewhat / moderately / strongly / very strongly), keeping text variance
    high while remaining natural language.

    Parameters
    ----------
    img : [3, H, W] float tensor in [0, 1]
    strength : [0, 1]  — controls how many style groups are activated
               (0 → none, 0.5 → 3 groups, 1.0 → all 6).
    rng : seeded RNG for reproducible style selection.
    jitter : [0, 1] — controls nuisance correlation between modalities.
             For each active style group, with probability ``jitter`` the
             caption describes an *independently sampled* style instead of
             the one actually applied to the image.  At 0 the correlation
             is perfect (caption matches image); at 1 the caption style is
             fully independent of the image style.
    jitter_seed : int — seed for the jitter RNG (typically sample index),
                  kept separate from ``rng`` to preserve backward-compatible
                  RNG stream for image transforms.

    Returns
    -------
    (styled_img, style_description)
        styled_img : [3, H, W] in [0, 1]
        style_description : e.g. "strongly sepia toned, moderately high
                            contrast, slightly grainy and noisy image of"
                            or "" if strength <= 0.
    """
    if strength <= 0:
        return img, ""

    n_groups = max(1, round(strength * N_STYLE_GROUPS))

    # Separate RNG for jitter draws so the main rng stream is identical
    # to the pre-jitter code (preserving backward compatibility at jitter=0).
    jitter_rng = np.random.RandomState(jitter_seed + 577215664)

    # Randomly select which groups to activate
    group_indices = list(range(N_STYLE_GROUPS))
    rng.shuffle(group_indices)
    active_groups = sorted(group_indices[:n_groups])

    descs: List[str] = []
    styled = img
    for gi in active_groups:
        group = STYLE_GROUPS[gi]
        choice = int(rng.randint(0, len(group)))
        style = group[choice]

        # Random continuous intensity ∈ [0.3, 1.0]
        t = float(rng.uniform(0.3, 1.0))
        styled = style["fn"](styled, t, rng)

        # Jitter: with probability `jitter`, describe a *different* style
        # from the same group.  Keeps the same intensity word so the only
        # change is the semantic style identity (not just an adjective).
        if jitter > 0 and jitter_rng.random() < jitter and len(group) > 1:
            # Pick a different style from this group
            alternatives = [i for i in range(len(group)) if i != choice]
            alt_choice = alternatives[
                int(jitter_rng.randint(0, len(alternatives)))
            ]
            descs.append(
                f"{_intensity_word(t)} {group[alt_choice]['desc']}"
            )
        else:
            descs.append(f"{_intensity_word(t)} {style['desc']}")

    style_text = ", ".join(descs) + " image of"
    return styled, style_text


# ═══════════════════════════════════════════════════════════════════════
# Option B: Scene Context
# ═══════════════════════════════════════════════════════════════════════

def load_scene_objects(
    instances_file: str,
    cat_names: List[str],
    cat_id_to_idx: Dict[int, int],
) -> Tuple[Dict[int, List[dict]], Dict[int, Tuple[int, int]]]:
    """
    Load *all* object annotations per image (not just the dominant one).

    Returns
    -------
    scene_objects : {image_id: [{name, bbox, area}, ...]} sorted by area desc
    image_dims : {image_id: (width, height)}
    """
    with open(instances_file) as f:
        data = json.load(f)

    image_dims: Dict[int, Tuple[int, int]] = {
        img["id"]: (img["width"], img["height"]) for img in data["images"]
    }

    raw: Dict[int, List[dict]] = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if cat_id not in cat_id_to_idx:
            continue
        raw.setdefault(img_id, []).append({
            "name": cat_names[cat_id_to_idx[cat_id]],
            "bbox": ann["bbox"],   # [x, y, w, h] in original pixels
            "area": ann["area"],
        })

    # Sort each image's objects by area descending
    scene_objects: Dict[int, List[dict]] = {}
    for img_id, objs in raw.items():
        scene_objects[img_id] = sorted(objs, key=lambda o: -o["area"])

    return scene_objects, image_dims


def format_scene_context(
    objects: List[dict],
    image_w: int,
    image_h: int,
    dominant_label_idx: int,
    cat_names: List[str],
    strength: float,
) -> str:
    """
    Generate a natural-language scene description from non-dominant objects.

    Parameters
    ----------
    objects : [{name, bbox, area}, ...] sorted by area descending.
    image_w, image_h : original image dimensions (for position normalisation).
    dominant_label_idx : category index of the dominant object.
    cat_names : list of category names.
    strength : [0, 1] — how many context categories to describe.

    Returns
    -------
    Description string, e.g.
    "in a scene with a large dining table in the center and 2 chairs on the left"
    or "" if nothing to describe.
    """
    if strength <= 0 or not objects:
        return ""

    dominant_name = cat_names[dominant_label_idx]

    # Group non-dominant objects by category (exclude dominant category)
    groups: Dict[str, List[dict]] = defaultdict(list)
    for obj in objects:
        if obj["name"] != dominant_name:
            groups[obj["name"]].append(obj)

    if not groups:
        return ""

    # Sort category groups by total area (most prominent first)
    sorted_groups = sorted(
        groups.items(), key=lambda kv: -sum(o["area"] for o in kv[1]),
    )

    # Select number of categories based on strength
    n = max(1, round(strength * min(len(sorted_groups), 6)))
    selected = sorted_groups[:n]

    parts: List[str] = []
    for name, objs in selected:
        # Use the largest instance for position
        largest = objs[0]
        x, y, w, h = largest["bbox"]
        cx = (x + w / 2) / max(image_w, 1)

        if cx < 0.33:
            h_pos = "on the left"
        elif cx > 0.67:
            h_pos = "on the right"
        else:
            h_pos = "in the center"

        rel_area = largest["area"] / max(image_w * image_h, 1)
        if rel_area > 0.15:
            size = "large"
        elif rel_area < 0.03:
            size = "small"
        else:
            size = ""

        count = len(objs)
        if count == 1:
            if size:
                parts.append(f"a {size} {name} {h_pos}")
            else:
                parts.append(f"a {name} {h_pos}")
        else:
            parts.append(f"{count} {name}s {h_pos}")

    if len(parts) == 1:
        return f"in a scene with {parts[0]}"
    return f"in a scene with {', '.join(parts[:-1])} and {parts[-1]}"


# Scene position/size words that might not hit min-freq in captions.
SCENE_VOCAB_TOKENS: List[str] = [
    "scene", "nearby", "foreground", "background",
]
