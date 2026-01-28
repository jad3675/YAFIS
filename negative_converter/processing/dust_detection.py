# Dust and scratch detection and removal
"""
Advanced detection and removal of dust spots and scratches from film scans.

Features:
- Local contrast analysis (not just global thresholds)
- Multi-scale detection
- Color channel correlation analysis
- Texture-aware confidence scoring
- Patch-based inpainting with texture matching
- Film grain preservation
"""

from typing import Tuple, List, Optional, NamedTuple
from dataclasses import dataclass, field
import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DustSpot(NamedTuple):
    """Detected dust spot."""
    x: int
    y: int
    radius: int
    confidence: float
    is_dark: bool = True  # Dark or bright spot


class Scratch(NamedTuple):
    """Detected scratch."""
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    confidence: float
    angle: float = 0.0  # Angle in radians


@dataclass
class DetectionResult:
    """Result of dust/scratch detection."""
    dust_spots: List[DustSpot]
    scratches: List[Scratch]
    mask: np.ndarray  # Binary mask of detected artifacts
    confidence_map: Optional[np.ndarray] = None  # Per-pixel confidence
    
    @property
    def total_artifacts(self) -> int:
        return len(self.dust_spots) + len(self.scratches)


@dataclass
class DetectionParams:
    """Parameters for dust/scratch detection."""
    # Dust detection
    dust_sensitivity: float = 0.5  # 0.0 to 1.0
    dust_min_size: int = 2  # Minimum dust spot size in pixels
    dust_max_size: int = 80  # Maximum dust spot size
    
    # Scratch detection
    scratch_sensitivity: float = 0.5
    scratch_min_length: int = 20  # Minimum scratch length
    scratch_max_width: int = 8  # Maximum scratch width
    
    # Local analysis
    local_window_size: int = 31  # Window for local statistics
    outlier_threshold: float = 2.5  # Standard deviations for outlier
    
    # Multi-scale
    num_scales: int = 3  # Number of scales to analyze
    
    # Texture awareness
    texture_threshold: float = 15.0  # Below this = smooth area
    smooth_area_boost: float = 1.5  # Confidence boost in smooth areas
    
    # Edge handling
    edge_margin: int = 5  # Ignore artifacts near edges
    
    # Channel correlation
    channel_correlation_threshold: float = 0.8  # High = likely dust



def _compute_local_stats(gray: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute local mean and standard deviation using efficient box filtering.
    
    Returns:
        Tuple of (local_mean, local_std)
    """
    import cv2
    
    # Use float64 for precision
    gray_f = gray.astype(np.float64)
    
    # Local mean using box filter
    local_mean = cv2.blur(gray_f, (window_size, window_size))
    
    # Local variance = E[X^2] - E[X]^2
    local_sq_mean = cv2.blur(gray_f ** 2, (window_size, window_size))
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 0)  # Numerical stability
    local_std = np.sqrt(local_var)
    
    return local_mean, local_std


def _detect_local_outliers(
    gray: np.ndarray,
    params: DetectionParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect pixels that are outliers relative to their local neighborhood.
    
    Returns:
        Tuple of (dark_outliers_mask, bright_outliers_mask)
    """
    local_mean, local_std = _compute_local_stats(gray, params.local_window_size)
    
    # Avoid division by zero in flat areas
    local_std = np.maximum(local_std, 1.0)
    
    # Z-score: how many std devs from local mean
    z_score = (gray.astype(np.float64) - local_mean) / local_std
    
    # Adjust threshold based on sensitivity
    threshold = params.outlier_threshold * (2.0 - params.dust_sensitivity)
    
    dark_outliers = z_score < -threshold
    bright_outliers = z_score > threshold
    
    return dark_outliers.astype(np.uint8), bright_outliers.astype(np.uint8)


def _compute_texture_map(gray: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    Compute local texture strength (variance-based).
    
    Low values = smooth areas (sky, solid colors)
    High values = textured areas (foliage, fabric)
    """
    _, local_std = _compute_local_stats(gray, window_size)
    return local_std


def _analyze_channel_correlation(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Analyze if detected spots appear similarly across all color channels.
    
    Dust typically affects all channels equally, while image features
    often have different intensities per channel.
    
    Returns:
        Confidence boost map (higher = more likely dust)
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return np.ones(mask.shape, dtype=np.float32)
    
    import cv2
    
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    
    # Compute local correlation between channels
    window = 11
    
    # Normalize channels locally
    r_mean = cv2.blur(r.astype(np.float32), (window, window))
    g_mean = cv2.blur(g.astype(np.float32), (window, window))
    b_mean = cv2.blur(b.astype(np.float32), (window, window))
    
    r_norm = r.astype(np.float32) - r_mean
    g_norm = g.astype(np.float32) - g_mean
    b_norm = b.astype(np.float32) - b_mean
    
    # Compute correlation coefficients
    eps = 1e-6
    r_std = np.sqrt(cv2.blur(r_norm ** 2, (window, window))) + eps
    g_std = np.sqrt(cv2.blur(g_norm ** 2, (window, window))) + eps
    b_std = np.sqrt(cv2.blur(b_norm ** 2, (window, window))) + eps
    
    # Correlation between R-G and R-B
    rg_corr = cv2.blur(r_norm * g_norm, (window, window)) / (r_std * g_std)
    rb_corr = cv2.blur(r_norm * b_norm, (window, window)) / (r_std * b_std)
    
    # High correlation = likely dust (affects all channels similarly)
    avg_corr = (np.abs(rg_corr) + np.abs(rb_corr)) / 2.0
    
    # Map to confidence boost (0.5 to 1.5)
    confidence_boost = 0.5 + avg_corr
    
    return confidence_boost


def _multi_scale_detection(
    gray: np.ndarray,
    params: DetectionParams
) -> List[np.ndarray]:
    """
    Detect outliers at multiple scales and combine results.
    
    Returns:
        List of detection masks at different scales.
    """
    import cv2
    
    masks = []
    current = gray.copy()
    
    for scale in range(params.num_scales):
        # Detect at current scale
        dark, bright = _detect_local_outliers(current, params)
        combined = cv2.bitwise_or(dark, bright)
        
        # Upscale mask back to original size if needed
        if scale > 0:
            combined = cv2.resize(
                combined, (gray.shape[1], gray.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        
        masks.append(combined)
        
        # Downsample for next scale
        if scale < params.num_scales - 1:
            current = cv2.pyrDown(current)
    
    return masks



def detect_dust_spots(
    image: np.ndarray,
    params: DetectionParams = None
) -> List[DustSpot]:
    """
    Detect dust spots using local contrast analysis and multi-scale detection.
    
    Args:
        image: Input image (RGB uint8).
        params: Detection parameters.
        
    Returns:
        List of detected dust spots.
    """
    if image is None:
        return []
    
    if params is None:
        params = DetectionParams()
    
    import cv2
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    spots = []
    
    # Multi-scale detection
    scale_masks = _multi_scale_detection(gray, params)
    
    # Combine masks (require detection at multiple scales for higher confidence)
    combined_mask = np.zeros_like(gray, dtype=np.float32)
    for i, mask in enumerate(scale_masks):
        weight = 1.0 / (i + 1)  # Higher weight for finer scales
        combined_mask += mask.astype(np.float32) * weight
    
    # Normalize and threshold
    combined_mask = combined_mask / len(scale_masks)
    artifact_mask = (combined_mask > 0.3).astype(np.uint8) * 255
    
    # Compute texture map for confidence adjustment
    texture_map = _compute_texture_map(gray, params.local_window_size // 2)
    
    # Channel correlation analysis
    channel_boost = _analyze_channel_correlation(image, artifact_mask)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    artifact_mask = cv2.morphologyEx(artifact_mask, cv2.MORPH_OPEN, kernel)
    artifact_mask = cv2.morphologyEx(artifact_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(artifact_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Detect if spots are dark or bright relative to surroundings
    local_mean, _ = _compute_local_stats(gray, params.local_window_size)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by size
        if area < params.dust_min_size ** 2:
            continue
        if area > params.dust_max_size ** 2:
            continue
        
        # Get bounding circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        cx, cy, radius = int(cx), int(cy), max(1, int(radius))
        
        # Skip if near edges
        if cx < params.edge_margin or cx > w - params.edge_margin:
            continue
        if cy < params.edge_margin or cy > h - params.edge_margin:
            continue
        
        # Calculate circularity (dust tends to be round)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0
        
        # Base confidence from circularity
        confidence = circularity * 0.6
        
        # Boost confidence in smooth areas (dust more visible/certain)
        local_texture = texture_map[cy, cx] if 0 <= cy < h and 0 <= cx < w else 50
        if local_texture < params.texture_threshold:
            confidence *= params.smooth_area_boost
        
        # Boost from channel correlation
        if 0 <= cy < h and 0 <= cx < w:
            confidence *= channel_boost[cy, cx]
        
        # Determine if dark or bright spot
        spot_mean = gray[max(0, cy-radius):min(h, cy+radius+1), 
                        max(0, cx-radius):min(w, cx+radius+1)].mean()
        local_mean_val = local_mean[cy, cx] if 0 <= cy < h and 0 <= cx < w else 128
        is_dark = spot_mean < local_mean_val
        
        confidence = min(1.0, max(0.0, confidence))
        
        if confidence > 0.25:
            spots.append(DustSpot(cx, cy, radius, confidence, is_dark))
    
    # Sort by confidence (highest first)
    spots.sort(key=lambda s: s.confidence, reverse=True)
    
    logger.debug("Detected %d dust spots", len(spots))
    return spots


def detect_scratches(
    image: np.ndarray,
    params: DetectionParams = None
) -> List[Scratch]:
    """
    Detect scratches using improved line detection with direction analysis.
    
    Args:
        image: Input image (RGB uint8).
        params: Detection parameters.
        
    Returns:
        List of detected scratches.
    """
    if image is None:
        return []
    
    if params is None:
        params = DetectionParams()
    
    import cv2
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    scratches = []
    
    # Detect local outliers for scratch detection
    dark_outliers, bright_outliers = _detect_local_outliers(gray, params)
    outlier_mask = cv2.bitwise_or(dark_outliers, bright_outliers) * 255
    
    # Morphological operations to connect broken lines
    # Use directional kernels for vertical and horizontal scratches
    kernels = [
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)),  # Vertical
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1)),  # Horizontal
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8),  # Diagonal
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8),  # Other diagonal
    ]
    
    all_lines = []
    
    for kernel in kernels:
        # Apply directional morphology
        enhanced = cv2.morphologyEx(outlier_mask, cv2.MORPH_CLOSE, kernel)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel)
        
        # Edge detection
        edges = cv2.Canny(enhanced, 30, 100, apertureSize=3)
        
        # Hough line detection
        threshold = max(20, int(40 * (1.0 - params.scratch_sensitivity)))
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=params.scratch_min_length,
            maxLineGap=15
        )
        
        if lines is not None:
            all_lines.extend(lines)
    
    # Process detected lines
    for line in all_lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate line properties
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        if length < params.scratch_min_length:
            continue
        
        # Calculate angle
        angle = np.arctan2(y2 - y1, x2 - x1)
        
        # Check if line is mostly vertical (common for film scratches)
        is_vertical = abs(abs(angle) - np.pi / 2) < np.pi / 6
        is_horizontal = abs(angle) < np.pi / 6 or abs(angle - np.pi) < np.pi / 6
        
        # Base confidence
        confidence = 0.4
        
        # Vertical scratches are more common on film
        if is_vertical:
            confidence += 0.3
        elif is_horizontal:
            confidence += 0.1
        
        # Longer lines are more likely scratches
        confidence += min(0.2, length / 300)
        
        # Check if the line follows outlier pixels
        # Sample points along the line
        num_samples = int(length / 5)
        outlier_count = 0
        for i in range(num_samples):
            t = i / max(1, num_samples - 1)
            px = int(x1 + t * (x2 - x1))
            py = int(y1 + t * (y2 - y1))
            if 0 <= px < w and 0 <= py < h:
                if outlier_mask[py, px] > 0:
                    outlier_count += 1
        
        outlier_ratio = outlier_count / max(1, num_samples)
        confidence *= (0.5 + outlier_ratio)
        
        confidence = min(1.0, max(0.0, confidence))
        
        if confidence > 0.35:
            scratches.append(Scratch(
                x1, y1, x2, y2,
                width=min(params.scratch_max_width, max(2, int(3 * confidence))),
                confidence=confidence,
                angle=angle
            ))
    
    # Remove duplicate/overlapping scratches
    scratches = _merge_overlapping_scratches(scratches)
    
    # Sort by confidence
    scratches.sort(key=lambda s: s.confidence, reverse=True)
    
    logger.debug("Detected %d scratches", len(scratches))
    return scratches


def _merge_overlapping_scratches(scratches: List[Scratch], threshold: float = 20.0) -> List[Scratch]:
    """Merge scratches that are very close or overlapping."""
    if len(scratches) <= 1:
        return scratches
    
    merged = []
    used = set()
    
    for i, s1 in enumerate(scratches):
        if i in used:
            continue
        
        # Find overlapping scratches
        group = [s1]
        for j, s2 in enumerate(scratches[i+1:], i+1):
            if j in used:
                continue
            
            # Check if scratches are close and parallel
            mid1 = ((s1.x1 + s1.x2) / 2, (s1.y1 + s1.y2) / 2)
            mid2 = ((s2.x1 + s2.x2) / 2, (s2.y1 + s2.y2) / 2)
            dist = np.sqrt((mid1[0] - mid2[0]) ** 2 + (mid1[1] - mid2[1]) ** 2)
            
            angle_diff = abs(s1.angle - s2.angle)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            
            if dist < threshold and angle_diff < np.pi / 6:
                group.append(s2)
                used.add(j)
        
        # Merge group into single scratch
        if len(group) == 1:
            merged.append(s1)
        else:
            # Use endpoints that maximize length
            all_points = []
            for s in group:
                all_points.extend([(s.x1, s.y1), (s.x2, s.y2)])
            
            # Find the two points that are furthest apart
            max_dist = 0
            best_pair = (all_points[0], all_points[1])
            for p1 in all_points:
                for p2 in all_points:
                    d = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                    if d > max_dist:
                        max_dist = d
                        best_pair = (p1, p2)
            
            avg_conf = sum(s.confidence for s in group) / len(group)
            avg_width = sum(s.width for s in group) // len(group)
            
            merged.append(Scratch(
                int(best_pair[0][0]), int(best_pair[0][1]),
                int(best_pair[1][0]), int(best_pair[1][1]),
                avg_width, avg_conf, s1.angle
            ))
        
        used.add(i)
    
    return merged



def create_artifact_mask(
    image: np.ndarray,
    dust_spots: List[DustSpot],
    scratches: List[Scratch],
    dilation: int = 2,
    feather: bool = True
) -> np.ndarray:
    """
    Create a binary mask of detected artifacts with optional feathering.
    
    Args:
        image: Input image for dimensions.
        dust_spots: List of detected dust spots.
        scratches: List of detected scratches.
        dilation: Pixels to dilate the mask.
        feather: Whether to feather edges for smoother inpainting.
        
    Returns:
        Mask (uint8, 0-255 with feathering or 0/255 without).
    """
    import cv2
    
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw dust spots with size based on confidence
    for spot in dust_spots:
        radius = spot.radius + dilation
        # Higher confidence = slightly larger mask
        radius = int(radius * (0.8 + spot.confidence * 0.4))
        cv2.circle(mask, (spot.x, spot.y), radius, 255, -1)
    
    # Draw scratches
    for scratch in scratches:
        width = scratch.width + dilation * 2
        cv2.line(
            mask,
            (scratch.x1, scratch.y1),
            (scratch.x2, scratch.y2),
            255,
            width
        )
    
    # Dilate mask
    if dilation > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation * 2 + 1, dilation * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Feather edges for smoother blending
    if feather:
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    return mask


def create_confidence_map(
    image: np.ndarray,
    dust_spots: List[DustSpot],
    scratches: List[Scratch]
) -> np.ndarray:
    """
    Create a per-pixel confidence map for detected artifacts.
    
    Returns:
        Float32 array with confidence values (0.0 to 1.0).
    """
    import cv2
    
    h, w = image.shape[:2]
    conf_map = np.zeros((h, w), dtype=np.float32)
    
    for spot in dust_spots:
        # Create circular gradient
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - spot.x) ** 2 + (y - spot.y) ** 2)
        spot_mask = dist <= spot.radius * 1.5
        
        # Confidence decreases from center
        spot_conf = spot.confidence * (1.0 - dist / (spot.radius * 1.5 + 1))
        spot_conf = np.clip(spot_conf, 0, spot.confidence)
        
        conf_map = np.maximum(conf_map, spot_conf * spot_mask)
    
    for scratch in scratches:
        # Create line with gradient
        # Simplified: just draw the line
        temp = np.zeros((h, w), dtype=np.float32)
        cv2.line(temp, (scratch.x1, scratch.y1), (scratch.x2, scratch.y2), 
                scratch.confidence, scratch.width)
        conf_map = np.maximum(conf_map, temp)
    
    return conf_map


def detect_artifacts(
    image: np.ndarray,
    params: DetectionParams = None
) -> DetectionResult:
    """
    Detect all artifacts (dust and scratches) in an image.
    
    Args:
        image: Input image (RGB uint8).
        params: Detection parameters.
        
    Returns:
        DetectionResult with all detected artifacts.
    """
    if params is None:
        params = DetectionParams()
    
    dust_spots = detect_dust_spots(image, params)
    scratches = detect_scratches(image, params)
    mask = create_artifact_mask(image, dust_spots, scratches)
    confidence_map = create_confidence_map(image, dust_spots, scratches)
    
    return DetectionResult(
        dust_spots=dust_spots,
        scratches=scratches,
        mask=mask,
        confidence_map=confidence_map
    )


def _find_similar_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 9,
    search_radius: int = 50
) -> np.ndarray:
    """
    Find similar patches for texture-aware inpainting.
    
    For each masked pixel, find the most similar patch in the surrounding area.
    """
    import cv2
    
    h, w = image.shape[:2]
    result = image.copy()
    
    # Get masked pixel coordinates
    masked_coords = np.where(mask > 127)
    if len(masked_coords[0]) == 0:
        return result
    
    half_patch = patch_size // 2
    
    # Convert to grayscale for matching
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Process in blocks for efficiency
    for y, x in zip(masked_coords[0], masked_coords[1]):
        if y < half_patch or y >= h - half_patch:
            continue
        if x < half_patch or x >= w - half_patch:
            continue
        
        # Define search region
        y_min = max(half_patch, y - search_radius)
        y_max = min(h - half_patch, y + search_radius)
        x_min = max(half_patch, x - search_radius)
        x_max = min(w - half_patch, x + search_radius)
        
        # Get target patch (with masked pixels)
        target = gray[y - half_patch:y + half_patch + 1, 
                     x - half_patch:x + half_patch + 1]
        
        best_match = None
        best_score = float('inf')
        
        # Search for best matching patch
        for sy in range(y_min, y_max, 3):  # Step for speed
            for sx in range(x_min, x_max, 3):
                # Skip if in masked region
                if mask[sy, sx] > 127:
                    continue
                
                candidate = gray[sy - half_patch:sy + half_patch + 1,
                               sx - half_patch:sx + half_patch + 1]
                
                if candidate.shape != target.shape:
                    continue
                
                # Compare only unmasked pixels
                patch_mask = mask[y - half_patch:y + half_patch + 1,
                                 x - half_patch:x + half_patch + 1]
                valid = patch_mask < 127
                
                if valid.sum() < patch_size:
                    continue
                
                diff = np.abs(target.astype(float) - candidate.astype(float))
                score = diff[valid].mean()
                
                if score < best_score:
                    best_score = score
                    best_match = (sy, sx)
        
        # Copy from best match
        if best_match is not None:
            sy, sx = best_match
            source_patch = image[sy - half_patch:sy + half_patch + 1,
                                sx - half_patch:sx + half_patch + 1]
            result[y, x] = source_patch[half_patch, half_patch]
    
    return result


def remove_artifacts(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "inpaint_telea",
    inpaint_radius: int = 5,
    preserve_grain: bool = True
) -> np.ndarray:
    """
    Remove artifacts using advanced inpainting with grain preservation.
    
    Args:
        image: Input image (RGB uint8).
        mask: Binary mask of artifacts to remove.
        method: Inpainting method.
        inpaint_radius: Radius for inpainting.
        preserve_grain: Whether to preserve film grain texture.
        
    Returns:
        Image with artifacts removed.
    """
    if image is None or mask is None:
        return image
    
    import cv2
    
    # Ensure mask is correct format
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    if mask.max() <= 1:
        mask = mask * 255
    
    # Threshold feathered mask for inpainting
    binary_mask = (mask > 50).astype(np.uint8) * 255
    
    # Auto-adjust inpaint radius based on artifact sizes
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        avg_size = np.sqrt(np.mean(areas)) if areas else 5
        inpaint_radius = max(3, min(15, int(avg_size / 2)))
    
    # Store grain pattern if preserving
    grain_pattern = None
    if preserve_grain:
        # Extract high-frequency component (grain)
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        grain_pattern = image.astype(np.float32) - blurred.astype(np.float32)
    
    # Perform inpainting
    if method == "inpaint_telea":
        result = cv2.inpaint(image, binary_mask, inpaint_radius, cv2.INPAINT_TELEA)
    elif method == "inpaint_ns":
        result = cv2.inpaint(image, binary_mask, inpaint_radius, cv2.INPAINT_NS)
    elif method == "patch_match":
        # Use our patch-based approach
        result = _find_similar_patches(image, binary_mask)
        # Blend with standard inpainting for robustness
        standard = cv2.inpaint(image, binary_mask, inpaint_radius, cv2.INPAINT_TELEA)
        alpha = 0.6
        result = (result.astype(float) * alpha + standard.astype(float) * (1 - alpha)).astype(np.uint8)
    elif method == "median":
        result = image.copy()
        kernel_size = inpaint_radius * 2 + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.medianBlur(image, kernel_size)
        result[binary_mask > 0] = blurred[binary_mask > 0]
    else:
        logger.warning("Unknown inpainting method: %s, using telea", method)
        result = cv2.inpaint(image, binary_mask, inpaint_radius, cv2.INPAINT_TELEA)
    
    # Restore grain pattern in inpainted areas
    if preserve_grain and grain_pattern is not None:
        # Sample grain from nearby non-masked areas
        dilated_mask = cv2.dilate(binary_mask, np.ones((15, 15), np.uint8))
        grain_sample_mask = (dilated_mask > 0) & (binary_mask == 0)
        
        if grain_sample_mask.sum() > 100:
            # Get grain statistics from nearby areas
            grain_std = grain_pattern[grain_sample_mask].std(axis=0)
            
            # Generate matching grain for inpainted areas
            inpaint_mask = binary_mask > 0
            noise = np.random.randn(*result.shape) * grain_std
            
            # Blend grain into result
            result_float = result.astype(np.float32)
            result_float[inpaint_mask] += noise[inpaint_mask] * 0.5
            result = np.clip(result_float, 0, 255).astype(np.uint8)
    
    # Smooth transition at mask edges using feathered mask
    if mask.max() > binary_mask.max():
        # Use original feathered mask for blending
        alpha = mask.astype(np.float32) / 255.0
        if len(image.shape) == 3:
            alpha = alpha[:, :, np.newaxis]
        result = (result.astype(np.float32) * alpha + 
                 image.astype(np.float32) * (1 - alpha))
        result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result




def auto_clean_image(
    image: np.ndarray,
    sensitivity: float = 0.5,
    inpaint_method: str = "inpaint_telea",
    preserve_grain: bool = True
) -> Tuple[np.ndarray, DetectionResult]:
    """
    Automatically detect and remove dust and scratches from an image.
    
    This is the main entry point for automatic artifact removal.
    
    Args:
        image: Input image (RGB uint8).
        sensitivity: Detection sensitivity (0.0 to 1.0).
        inpaint_method: Method for inpainting ("inpaint_telea", "inpaint_ns", 
                       "patch_match", "median").
        preserve_grain: Whether to preserve film grain texture.
        
    Returns:
        Tuple of (cleaned_image, detection_result).
    """
    if image is None:
        return image, DetectionResult([], [], np.zeros((1, 1), dtype=np.uint8))
    
    # Create detection parameters from sensitivity
    params = DetectionParams(
        dust_sensitivity=sensitivity,
        scratch_sensitivity=sensitivity,
        # Adjust thresholds based on sensitivity
        outlier_threshold=3.0 - sensitivity * 1.5,  # 1.5 to 3.0
        texture_threshold=10.0 + (1.0 - sensitivity) * 20.0,  # 10 to 30
    )
    
    # Detect artifacts
    result = detect_artifacts(image, params)
    
    logger.info(
        "Auto-clean detected %d dust spots and %d scratches",
        len(result.dust_spots), len(result.scratches)
    )
    
    # Remove artifacts if any found
    if result.total_artifacts > 0:
        # Calculate appropriate inpaint radius
        if result.dust_spots:
            avg_radius = sum(s.radius for s in result.dust_spots) / len(result.dust_spots)
            inpaint_radius = max(3, min(12, int(avg_radius * 1.5)))
        else:
            inpaint_radius = 5
        
        cleaned = remove_artifacts(
            image, 
            result.mask,
            method=inpaint_method,
            inpaint_radius=inpaint_radius,
            preserve_grain=preserve_grain
        )
    else:
        cleaned = image.copy()
    
    return cleaned, result


def preview_detection(
    image: np.ndarray,
    sensitivity: float = 0.5,
    highlight_color: Tuple[int, int, int] = (255, 0, 0),
    overlay_alpha: float = 0.5
) -> Tuple[np.ndarray, DetectionResult]:
    """
    Create a preview image showing detected artifacts highlighted.
    
    Useful for showing users what will be cleaned before applying.
    
    Args:
        image: Input image (RGB uint8).
        sensitivity: Detection sensitivity (0.0 to 1.0).
        highlight_color: RGB color for highlighting artifacts.
        overlay_alpha: Opacity of the highlight overlay.
        
    Returns:
        Tuple of (preview_image, detection_result).
    """
    import cv2
    
    if image is None:
        return image, DetectionResult([], [], np.zeros((1, 1), dtype=np.uint8))
    
    # Create detection parameters
    params = DetectionParams(
        dust_sensitivity=sensitivity,
        scratch_sensitivity=sensitivity,
        outlier_threshold=3.0 - sensitivity * 1.5,
        texture_threshold=10.0 + (1.0 - sensitivity) * 20.0,
    )
    
    # Detect artifacts
    result = detect_artifacts(image, params)
    
    # Create preview image
    preview = image.copy()
    
    # Create colored overlay
    overlay = np.zeros_like(image)
    
    # Draw dust spots
    for spot in result.dust_spots:
        # Draw filled circle with confidence-based radius
        radius = int(spot.radius * (1.0 + spot.confidence * 0.5))
        cv2.circle(overlay, (spot.x, spot.y), radius, highlight_color, -1)
        # Draw outline
        cv2.circle(preview, (spot.x, spot.y), radius + 2, (255, 255, 0), 1)
    
    # Draw scratches
    for scratch in result.scratches:
        width = scratch.width + 2
        cv2.line(overlay, (scratch.x1, scratch.y1), (scratch.x2, scratch.y2),
                highlight_color, width)
        # Draw outline
        cv2.line(preview, (scratch.x1, scratch.y1), (scratch.x2, scratch.y2),
                (255, 255, 0), 1)
    
    # Blend overlay with preview
    mask = (overlay.sum(axis=2) > 0).astype(np.float32)[:, :, np.newaxis]
    preview = (preview.astype(np.float32) * (1 - mask * overlay_alpha) + 
              overlay.astype(np.float32) * mask * overlay_alpha)
    preview = np.clip(preview, 0, 255).astype(np.uint8)
    
    logger.debug(
        "Preview shows %d dust spots and %d scratches",
        len(result.dust_spots), len(result.scratches)
    )
    
    return preview, result
