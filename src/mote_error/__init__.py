"""
MOTe error: Missing, Overlap, Trespass, and the support metric Excess

A library for evaluating document layout analysis models with the MOTe error,
and page-level OCR with the Character Error Vector.
"""

__version__ = "0.1.1"

from .layout import mote, missing, coverage, overlap, trespass, excess, iou, mean_iou
from .ocr import (
    shannon_entropy,
    jensen_shannon_divergence,
    text_to_counter,
    jsd_distance,
    spacer,
    spacer_micro,
    cdd_decomp,
    spacer_decomp,
    cdd_decomp_spatial,
    spacer_decomp_spatial,
)
from .class_metrics import (
    mote_class,
    coverage_matrix,
    overlap_matrix,
    trespass_matrix,
    class_confusion_counts,
    sum_class_counts,
    finalize_class_counts,
)
from .types import MOTeResult, ClassMOTeResult, ClassMOTeCounts, TokenPositions, RegionChars, RegionPixels, GTBoxes, CDDDecomposition, SpACERDecomposition
from .adapters import boxes_to_region_pixels, polygons_to_panoptic_mask
from .panoptic_quality import panoptic_quality
from .visualisation import (
    MIXED_KEY,
    class_palette,
    compute_class_masks,
    compute_mote_masks,
    visualize_class_masks,
    visualize_mote_states,
)
from .dataset import (
    load_limerick_example,
    extract_ssu_boxes,
    extract_line_boxes,
    extract_word_boxes,
    chars_to_region_chars,
    reconstruct_text,
)
from .alto_ssu_tagger import ALTOSSUTagger, assign_alto_ssu

__all__ = [
    "mote",
    "missing",
    "coverage",
    "overlap",
    "trespass",
    "excess",
    "iou",
    "mean_iou",
    "shannon_entropy",
    "jensen_shannon_divergence",
    "text_to_counter",
    "jsd_distance",
    "spacer",
    "spacer_micro",
    "cdd_decomp",
    "spacer_decomp",
    "cdd_decomp_spatial",
    "spacer_decomp_spatial",
    "mote_class",
    "coverage_matrix",
    "overlap_matrix",
    "trespass_matrix",
    "class_confusion_counts",
    "sum_class_counts",
    "finalize_class_counts",
    "MOTeResult",
    "ClassMOTeResult",
    "ClassMOTeCounts",
    "TokenPositions",
    "RegionChars",
    "RegionPixels",
    "GTBoxes",
    "CDDDecomposition",
    "SpACERDecomposition",
    "boxes_to_region_pixels",
    "polygons_to_panoptic_mask",
    "panoptic_quality",
    "compute_mote_masks",
    "visualize_mote_states",
    "compute_class_masks",
    "visualize_class_masks",
    "class_palette",
    "MIXED_KEY",
    "load_limerick_example",
    "extract_ssu_boxes",
    "extract_line_boxes",
    "extract_word_boxes",
    "chars_to_region_chars",
    "reconstruct_text",
    "ALTOSSUTagger",
    "assign_alto_ssu",
]
