"""Run DermEval inference with a compatible local LLaVA-style runtime.

This module intentionally keeps the heavy multimodal runtime optional. Importing
``dermbench.dermeval`` does not require torch, transformers, or LLaVA; those
dependencies are only needed when this CLI is executed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from io import BytesIO
from typing import Any, Dict, Optional

try:
    import requests
    import torch
    from PIL import Image

    from llava.constants import (
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
        IMAGE_PLACEHOLDER,
        IMAGE_TOKEN_INDEX,
    )
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
except ImportError as exc:  # pragma: no cover - depends on optional runtime.
    requests = None
    torch = None
    Image = None
    DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IM_END_TOKEN = "</im_end>"
    DEFAULT_IM_START_TOKEN = "<im_start>"
    IMAGE_PLACEHOLDER = "<image-placeholder>"
    IMAGE_TOKEN_INDEX = -200
    conv_templates = None
    get_model_name_from_path = None
    process_images = None
    tokenizer_image_token = None
    load_pretrained_model = None
    disable_torch_init = None
    _OPTIONAL_IMPORT_ERROR = exc
else:
    _OPTIONAL_IMPORT_ERROR = None

from .prompts import build_dermeval_prompt
from .score_parser import parse_scores


def _require_llava_runtime() -> None:
    if _OPTIONAL_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "DermEval local inference requires an installed LLaVA-compatible runtime "
        "plus torch, Pillow, and requests. Install those optional dependencies and "
        "provide a compatible DermEval/LLaVA checkpoint before running this command."
    ) from _OPTIONAL_IMPORT_ERROR


def load_image(image_file: str) -> Image.Image:
    _require_llava_runtime()
    if image_file.startswith("http://") or image_file.startswith("https://"):
        response = requests.get(image_file, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")


def infer_conv_mode(model_name: str) -> str:
    name = model_name.lower()
    if "llama-2" in name:
        return "llava_llama_2"
    if "mistral" in name:
        return "mistral_instruct"
    if "v1.6-34b" in name:
        return "chatml_direct"
    if "v1" in name:
        return "llava_v1"
    if "mpt" in name:
        return "mpt"
    return "llava_v0"


def build_prompt_for_llava(query: str, model: Any, conv_mode: str) -> str:
    _require_llava_runtime()
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in query:
        query = re.sub(IMAGE_PLACEHOLDER, image_token_se if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN, query)
    else:
        query = (image_token_se if model.config.mm_use_im_start_end else DEFAULT_IMAGE_TOKEN) + "\n" + query
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def run_one(args: argparse.Namespace) -> Dict[str, Any]:
    _require_llava_runtime()
    disable_torch_init()
    model_name = args.model_name or get_model_name_from_path(args.model_path)
    # DermEval checkpoints may be named "dermeval-*" even though they are LLaVA models.
    # LLaVA's loader branches on the model name string, so force the LLaVA path.
    if "llava" not in model_name.lower():
        model_name = "llava-" + model_name
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        args.model_base,
        model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device=args.device,
        use_flash_attn=args.use_flash_attn,
    )
    conv_mode = args.conv_mode or infer_conv_mode(model_name)
    diagnostic_text = args.diagnostic_text
    if args.diagnostic_text_file:
        with open(args.diagnostic_text_file, "r", encoding="utf-8") as f:
            diagnostic_text = f.read().strip()
    if not diagnostic_text:
        raise ValueError("Provide --diagnostic-text or --diagnostic-text-file.")

    query = build_dermeval_prompt(diagnostic_text)
    prompt = build_prompt_for_llava(query, model, conv_mode)
    image = load_image(args.image_file)
    image_sizes = [image.size]
    image_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    result = {
        "image": args.image_file,
        "diagnostic_text": diagnostic_text,
        "evaluation_text": output_text,
        "scores": parse_scores(output_text),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="DermEval single-case inference.")
    parser.add_argument("--model-path", required=True, help="DermEval/LLaVA model or LoRA checkpoint path.")
    parser.add_argument("--model-base", default=None, help="Base model path if --model-path is a LoRA adapter.")
    parser.add_argument("--image-file", required=True, help="Dermatology image file or URL.")
    parser.add_argument("--diagnostic-text", default=None, help="Candidate diagnostic narrative.")
    parser.add_argument("--diagnostic-text-file", default=None, help="File containing candidate diagnostic narrative.")
    parser.add_argument("--conv-mode", default=None, help="Conversation template, e.g. llava_v1.")
    parser.add_argument("--model-name", default=None, help="Optional loader name override; useful for DermEval checkpoints.")
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or mps.")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--output-json", default=None, help="Optional path to save JSON output.")
    args = parser.parse_args()

    result = run_one(args)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
