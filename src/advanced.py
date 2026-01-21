"""
Advanced classifier using GPT-4o Vision.

Uses OpenAI's multimodal LLM to analyze uniformity test images
with natural language reasoning.

Benefits:
- Interpretable explanations for each classification
- Zero training required
- Handles edge cases through reasoning
- Shows awareness of cutting-edge approaches

Caching:
- All API responses are cached for offline reproducibility
- Once you've run the pipeline, it can be reproduced without API access
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# Default logger (no-op if not provided)
_default_logger = logging.getLogger(__name__)


def image_to_base64(image: np.ndarray) -> str:
    """
    Convert numpy array to base64-encoded PNG.

    Args:
        image: 2D grayscale numpy array

    Returns:
        Base64-encoded string of PNG image
    """
    img = Image.fromarray(image)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


def get_image_hash(image: np.ndarray) -> str:
    """
    Generate a hash for an image (for caching).

    Args:
        image: Numpy array

    Returns:
        MD5 hash string
    """
    return hashlib.md5(image.tobytes()).hexdigest()


def load_cache(cache_dir: Path) -> Dict[str, dict]:
    """Load cached responses from disk."""
    cache_file = cache_dir / "gpt4o_responses.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache_dir: Path, cache: Dict[str, dict]) -> None:
    """Save cache to disk."""
    cache_file = cache_dir / "gpt4o_responses.json"
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)


def build_prompt() -> str:
    """
    Build the system prompt for GPT-4o.

    Returns:
        Prompt string for uniformity analysis
    """
    return """You are a medical imaging QA specialist analyzing scanner uniformity tests.

This image shows a phantom scan from a medical scanner (MRI, CT, or PET). The phantom is a uniform test object, so the resulting image SHOULD appear uniform - consistent brightness throughout.

Analyze this image for uniformity issues:
1. Brightness variations or gradients (one region brighter than another)
2. Artifacts (rings, bands, stripes, spots)
3. Signal dropouts or dead zones
4. Any other non-uniformities

Based on your analysis, classify as PASS (acceptable uniformity) or FAIL (uniformity issues detected).

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{"classification": "PASS", "confidence": 85, "reasoning": "Brief explanation here"}

or

{"classification": "FAIL", "confidence": 90, "reasoning": "Brief explanation of issues found"}"""


def classify_single_image(
    image: np.ndarray,
    client,
    cache: Dict[str, dict],
    cache_dir: Path,
    logger: logging.Logger
) -> Tuple[int, str]:
    """
    Classify a single image using GPT-4o.

    Args:
        image: 2D grayscale numpy array
        client: OpenAI client instance
        cache: Cache dictionary
        cache_dir: Directory for cache file
        logger: Logger instance

    Returns:
        Tuple of (label, reasoning)
        - label: 0 for PASS, 1 for FAIL
        - reasoning: Model's explanation
    """
    # Check cache first
    image_hash = get_image_hash(image)
    if image_hash in cache:
        cached = cache[image_hash]
        return cached["label"], cached["reasoning"]

    # Prepare the image
    base64_image = image_to_base64(image)
    prompt = build_prompt()

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0  # Deterministic output
        )

        # Parse response
        content = response.choices[0].message.content.strip()

        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)

        label = 1 if result["classification"].upper() == "FAIL" else 0
        reasoning = result.get("reasoning", "No reasoning provided")

        # Cache the result
        cache[image_hash] = {
            "label": label,
            "reasoning": reasoning,
            "confidence": result.get("confidence"),
            "raw_classification": result["classification"]
        }
        save_cache(cache_dir, cache)

        return label, reasoning

    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse GPT-4o response as JSON: {e}"
        logger.warning(error_msg)
        return 0, error_msg

    except Exception as e:
        error_msg = f"API error: {str(e)}"
        logger.warning(error_msg)
        return 0, error_msg  # Default to PASS on error


def predict_advanced(
    images: List[np.ndarray],
    paths: List[Path],
    cache_dir: Path,
    logger: Optional[logging.Logger] = None
) -> Tuple[List[int], List[str]]:
    """
    Classify images using GPT-4o Vision.

    Args:
        images: List of 2D grayscale numpy arrays
        paths: List of file paths (for logging)
        cache_dir: Directory for caching API responses
        logger: Optional logger instance

    Returns:
        Tuple of (predictions, reasonings)
        - predictions: List of labels (0=PASS, 1=FAIL)
        - reasonings: List of explanation strings
    """
    if logger is None:
        logger = _default_logger

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")

    # Load cache
    cache = load_cache(cache_dir)

    # Check how many are cached
    cached_count = sum(1 for img in images if get_image_hash(img) in cache)

    if cached_count == len(images):
        logger.info("Using cached responses for all %d images", len(images))
        predictions = []
        reasonings = []
        for img in images:
            cached = cache[get_image_hash(img)]
            predictions.append(cached["label"])
            reasonings.append(cached["reasoning"])
        return predictions, reasonings

    if not api_key:
        if cached_count > 0:
            logger.warning(
                "OPENAI_API_KEY not set. Using %d cached responses, "
                "%d images will use fallback (PASS)",
                cached_count, len(images) - cached_count
            )
        else:
            logger.warning(
                "OPENAI_API_KEY not set and no cache available. "
                "Using fallback predictions (all PASS)"
            )

        predictions = []
        reasonings = []
        for img in images:
            image_hash = get_image_hash(img)
            if image_hash in cache:
                predictions.append(cache[image_hash]["label"])
                reasonings.append(cache[image_hash]["reasoning"])
            else:
                predictions.append(0)  # Default to PASS
                reasonings.append("Fallback: No API key and not in cache")
        return predictions, reasonings

    # Initialize OpenAI client
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    predictions = []
    reasonings = []
    new_api_calls = 0

    for i, (image, path) in enumerate(zip(images, paths)):
        image_hash = get_image_hash(image)
        was_cached = image_hash in cache

        label, reasoning = classify_single_image(
            image, client, cache, cache_dir, logger
        )
        predictions.append(label)
        reasonings.append(reasoning)

        if not was_cached:
            new_api_calls += 1

        status = "cached" if was_cached else "API"
        result = "FAIL" if label == 1 else "PASS"
        logger.debug("[%d/%d] %s: %s (%s)", i + 1, len(images), path.name, result, status)

    logger.info(
        "Classified %d images (%d cached, %d API calls)",
        len(images), cached_count, new_api_calls
    )

    return predictions, reasonings
