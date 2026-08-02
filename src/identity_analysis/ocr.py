#!/usr/bin/env python3
"""ONNX line recognition and CTC decoding."""
import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from .onnx_runtime import create_session


DEFAULT_MODEL_NAME = "models/ocr_latin.onnx"
DEFAULT_CHARSET_NAME = "charsets/latin.txt"
INPUT_HEIGHT = 48


@lru_cache(maxsize=4)
def ocr_locale_catalog(resource: Path) -> dict:
    metadata_path = resource / "metadata" / "ocr_scripts.json"
    if not metadata_path.is_file():
        return {"localeToModel": {}, "models": {}}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    locale_to_model = {}
    models = {}
    for definition in metadata.get("TextRecognizer", []):
        match = re.search(r"(\d+)$", definition.get("netName", ""))
        if not match:
            continue
        model_locale = int(match.group(1))
        model = resource / "models" / "ocr" / f"{model_locale}.onnx"
        charset = resource / "charsets" / "ocr" / f"{model_locale}.txt"
        if not model.is_file() or not charset.is_file():
            continue
        models[model_locale] = {
            "name": definition.get("netName"),
            "model": model,
            "charset": charset,
            "locales": definition.get("lcids", []),
        }
        for locale in definition.get("lcids", []):
            locale_to_model[int(locale)] = model_locale
    return {"localeToModel": locale_to_model, "models": models}


def resolve_ocr_locale(resource: Path, locale: int | str | None) -> int:
    try:
        locale_number = int(locale) if locale is not None else 0
    except (TypeError, ValueError):
        locale_number = 0
    return ocr_locale_catalog(resource.resolve())["localeToModel"].get(
        locale_number, 0
    )


@lru_cache(maxsize=16)
def load_assets(resource: Path, locale: int | str | None = None) -> tuple[bytes, tuple[str, ...]]:
    if not resource.is_dir():
        raise ValueError(f"Expected a prepared asset directory: {resource}")
    model_locale = resolve_ocr_locale(resource, locale)
    if model_locale:
        definition = ocr_locale_catalog(resource)["models"][model_locale]
        model_path = definition["model"]
        charset_path = definition["charset"]
    else:
        model_path = resource / DEFAULT_MODEL_NAME
        charset_path = resource / DEFAULT_CHARSET_NAME
    model = model_path.read_bytes()
    charset_codes = charset_path.read_text(encoding="ascii").splitlines()
    charset = tuple(chr(int(code)) for code in charset_codes if code.strip())
    return model, charset


@lru_cache(maxsize=16)
def load_runtime(
    resource: Path, locale: int | str | None = None
) -> tuple[ort.InferenceSession, tuple[str, ...]]:
    model, charset = load_assets(resource, locale)
    session = create_session(model)
    return session, charset


def prepare_image(source: Path | Image.Image, invert: bool, normalization: str) -> np.ndarray:
    image = Image.open(source).convert("L") if isinstance(source, Path) else source.convert("L")
    if invert:
        image = ImageOps.invert(image)
    width = max(1, round(image.width * INPUT_HEIGHT / image.height))
    image = image.resize((width, INPUT_HEIGHT), Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.float32)
    if normalization == "zero-one":
        values /= 255.0
    elif normalization == "minus-one-one":
        values = values / 127.5 - 1.0
    return values[np.newaxis, np.newaxis, :, :]


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def decode_ctc(logits: np.ndarray, charset: tuple[str, ...]) -> tuple[str, float, list[int]]:
    probabilities = softmax(logits[0])
    classes = np.argmax(probabilities, axis=-1)
    confidences = np.max(probabilities, axis=-1)
    decoded = []
    decoded_confidences = []
    previous = None
    blank_index = len(charset)
    for class_index, confidence in zip(classes.tolist(), confidences.tolist()):
        if class_index != previous and class_index != blank_index:
            if 0 <= class_index < len(charset):
                decoded.append(charset[class_index])
                decoded_confidences.append(confidence)
        previous = class_index
    mean_confidence = (
        float(np.mean(decoded_confidences)) if decoded_confidences else 0.0
    )
    return "".join(decoded), mean_confidence, classes.tolist()


def run(
    resource: Path,
    image: Path | Image.Image,
    invert: bool,
    normalization: str,
    locale: int | str | None = None,
) -> dict:
    resource = resource.resolve()
    model_locale = resolve_ocr_locale(resource, locale)
    session, charset = load_runtime(resource, locale)
    tensor = prepare_image(image, invert, normalization)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits = session.run([output_name], {input_name: tensor})[0]
    text, confidence, raw_classes = decode_ctc(logits, charset)
    return {
        "text": text,
        "confidence": confidence,
        "input_shape": list(tensor.shape),
        "output_shape": list(logits.shape),
        "normalization": normalization,
        "inverted": invert,
        "raw_classes": raw_classes,
        "requestedLocale": locale,
        "modelLocale": model_locale,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the Latin line-recognition ONNX model from an asset directory."
    )
    parser.add_argument("resource", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--locale", type=int, help="Field LCID used to select the OCR model")
    parser.add_argument(
        "--normalization",
        choices=("none", "zero-one", "minus-one-one"),
        default="minus-one-one",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.resource,
                args.image,
                args.invert,
                args.normalization,
                locale=args.locale,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
