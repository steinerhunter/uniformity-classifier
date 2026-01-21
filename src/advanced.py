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

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import io


def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy array to base64-encoded PNG."""
    img = Image.fromarray(image)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


def get_image_hash(image: np.ndarray) -> str:
    """Generate a hash for an image (for caching)."""
    return hashlib.md5(image.tobytes()).hexdigest()


def load_cache(cache_dir: Path) -> dict:
    """Load cached responses from disk."""
    cache_file = cache_dir / "gpt4o_responses.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache_dir: Path, cache: dict):
    """Save cache to disk."""
    cache_file = cache_dir / "gpt4o_responses.json"
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)


def classify_single_image(
    image: np.ndarray,
    client,
    cache: dict,
    cache_dir: Path
) -> Tuple[int, str]:
    """
    Classify a single image using GPT-4o.

    Args:
        image: 2D grayscale numpy array
        client: OpenAI client
        cache: Cache dictionary
        cache_dir: Directory for cache file

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

    # Create the prompt
    prompt = """You are a medical imaging QA specialist analyzing scanner uniformity tests.

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
        cache[image_hash] = {"label": label, "reasoning": reasoning}
        save_cache(cache_dir, cache)

        return label, reasoning

    except Exception as e:
        # On error, return uncertain prediction with error message
        error_msg = f"API error: {str(e)}"
        print(f"      Warning: {error_msg}")
        return 0, error_msg  # Default to PASS on error


def predict_advanced(
    images: List[np.ndarray],
    paths: List[Path],
    cache_dir: Path
) -> Tuple[List[int], List[str]]:
    """
    Classify images using GPT-4o Vision.

    Args:
        images: List of 2D grayscale numpy arrays
        paths: List of file paths (for logging)
        cache_dir: Directory for caching API responses

    Returns:
        Tuple of (predictions, reasonings)
        - predictions: List of labels (0=PASS, 1=FAIL)
        - reasonings: List of explanation strings
    """
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")

    # Load cache
    cache = load_cache(cache_dir)

    # Check how many are cached
    cached_count = sum(1 for img in images if get_image_hash(img) in cache)

    if cached_count == len(images):
        print(f"      Using cached responses for all {len(images)} images")
        predictions = []
        reasonings = []
        for img in images:
            cached = cache[get_image_hash(img)]
            predictions.append(cached["label"])
            reasonings.append(cached["reasoning"])
        return predictions, reasonings

    if not api_key:
        if cached_count > 0:
            print(f"      Warning: OPENAI_API_KEY not set. Using {cached_count} cached responses.")
            print(f"      {len(images) - cached_count} images will use fallback (PASS).")
        else:
            print("      Warning: OPENAI_API_KEY not set and no cache available.")
            print("      Using fallback predictions (all PASS).")

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

    for i, (image, path) in enumerate(zip(images, paths)):
        image_hash = get_image_hash(image)
        cached = image_hash in cache

        label, reasoning = classify_single_image(image, client, cache, cache_dir)
        predictions.append(label)
        reasonings.append(reasoning)

        status = "cached" if cached else "API"
        result = "FAIL" if label == 1 else "PASS"
        print(f"      [{i + 1}/{len(images)}] {path.name}: {result} ({status})")

    return predictions, reasonings
