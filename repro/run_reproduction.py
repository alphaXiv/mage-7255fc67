#!/usr/bin/env python
"""Bounded Mage-VL codec-vs-frame reproduction with log-complete evidence."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import re
import statistics
import subprocess
import time
import urllib.request
from queue import Empty
from pathlib import Path


DATA_DIR = Path(".cache/perception_test")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def emit(kind: str, payload: dict) -> None:
    print(f"{kind}={json.dumps(payload, sort_keys=True)}", flush=True)


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, temp)
    temp.replace(path)


def extract_zip(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["bsdtar", "-xf", str(path), "-C", str(destination)],
        check=True,
    )


def prepare_data(config: dict) -> tuple[Path, dict]:
    videos_zip = DATA_DIR / "sample_videos.zip"
    annotations_zip = DATA_DIR / "sample_annotations.zip"
    download(config["public_data"]["videos_url"], videos_zip)
    download(config["public_data"]["annotations_url"], annotations_zip)
    extracted = DATA_DIR / "extracted"
    annotations_path = extracted / "sample.json"
    if not annotations_path.exists():
        extract_zip(annotations_zip, extracted)
    wanted = set(config["video_ids"])
    found = {p.stem for p in extracted.rglob("*.mp4")}
    if not wanted.issubset(found):
        extract_zip(videos_zip, extracted)
    annotations = json.loads(annotations_path.read_text())
    return extracted, annotations


def locate_video(root: Path, video_id: str) -> Path:
    matches = list(root.rglob(f"{video_id}.mp4"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one video for {video_id}, found {matches}")
    return matches[0]


def build_prompt(processor, question: dict) -> str:
    options = "\n".join(
        f"{LETTERS[i]}. {value}" for i, value in enumerate(question["options"])
    )
    prompt = (
        "Answer the multiple-choice question using the video. "
        "Return only the option letter.\n"
        f"Question: {question['question']}\nOptions:\n{options}"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def parse_choice(text: str, option_count: int) -> int | None:
    text = text.strip().upper()
    patterns = [
        r"^\s*[\(\[]?([A-Z])[\)\].,:;\s]?",
        r"\bOPTION\s+([A-Z])\b",
        r"\bANSWER\s*(?:IS|:)?\s*([A-Z])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            index = ord(match.group(1)) - ord("A")
            if 0 <= index < option_count:
                return index
    return None


def visual_token_count(inputs: dict, merge_size: int) -> int:
    grid = inputs["image_grid_thw"]
    return int((grid.prod(dim=-1) // (merge_size * merge_size)).sum().item())


def source_frame_count(video_path: Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return count


def method_inputs(processor, prompt: str, video: Path, method: dict, config: dict):
    common = {
        "text": [prompt],
        "videos": [str(video)],
        "return_tensors": "pt",
        "padding": True,
    }
    if method["backend"] == "codec":
        codec_config = {
            "engine": method.get("codec_engine", "hevc"),
            "target_canvas": int(method["budget"]),
            "patch": 16,
        }
        if codec_config["engine"] == "dcvc-rt":
            codec_config["dcvc"] = {
                "pkg_dir": str(Path(config["_checkpoint_path"]) / "neural_codec"),
                "device": config["_device"],
            }
        return processor(
            **common,
            video_backend="codec",
            max_pixels=int(config["max_pixels"]),
            codec_config=codec_config,
        )
    return processor(
        **common,
        video_backend="frames",
        num_frames=int(method["budget"]),
    )


def timed_generate(model, processor, inputs: dict, device: str, repeats: int, max_new: int):
    import torch

    on_device = {
        key: (
            value.to(device=device, dtype=model.dtype)
            if key == "pixel_values"
            else value.to(device)
        )
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        model.generate(**on_device, max_new_tokens=max_new, do_sample=False)
    torch.cuda.synchronize(device)
    durations = []
    texts = []
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **on_device, max_new_tokens=max_new, do_sample=False
            )
        torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)
        texts.append(
            processor.tokenizer.decode(
                output[0, on_device["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ).strip()
        )
    peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    return durations, texts, peak_gib


def worker(
    worker_index: int,
    tasks: list[dict],
    config: dict,
    annotations: dict,
    data_root: str,
    output: mp.Queue,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = f"cuda:{worker_index}"
    torch.cuda.set_device(worker_index)
    random.seed(int(config["seed"]) + worker_index)
    processor = AutoProcessor.from_pretrained(
        config["_checkpoint_path"], trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["_checkpoint_path"],
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    merge_size = int(getattr(processor, "spatial_merge_size", 2))
    for task in tasks:
        method = task["method"]
        video_id = task["video_id"]
        video = locate_video(Path(data_root), video_id)
        questions = annotations[video_id]["mc_question"]
        total_frames = source_frame_count(video)
        local_config = dict(config, _device=device)
        for q_index, question in enumerate(questions):
            prompt = build_prompt(processor, question)
            pre_start = time.perf_counter()
            inputs = method_inputs(
                processor, prompt, video, method, local_config
            )
            pre_seconds = time.perf_counter() - pre_start
            retained = visual_token_count(inputs, merge_size)
            canvas_count = int(inputs["image_grid_thw"].shape[0])
            nominal_frames = min(total_frames, int(method["budget"]) * 8)
            if method["backend"] == "frames":
                actual_frames = canvas_count
                per_frame = retained / max(actual_frames, 1)
            else:
                per_frame = retained / max(int(method["budget"]), 1)
            dense_duration_tokens = int(round(per_frame * nominal_frames))
            durations, texts, peak_gib = timed_generate(
                model,
                processor,
                inputs,
                device,
                int(config["timing_repeats"]),
                int(config["max_new_tokens"]),
            )
            prediction = parse_choice(texts[0], len(question["options"]))
            record = {
                "experiment": config["experiment"],
                "worker": worker_index,
                "video_id": video_id,
                "question_id": int(question["id"]),
                "area": question.get("area"),
                "reasoning": question.get("reasoning"),
                "method": method["name"],
                "backend": method["backend"],
                "budget": int(method["budget"]),
                "source_video_frames": total_frames,
                "matched_duration_sampled_frames": nominal_frames,
                "canvas_count": canvas_count,
                "retained_visual_tokens": retained,
                "dense_duration_reference_tokens": dense_duration_tokens,
                "token_reduction_fraction": (
                    1.0 - retained / dense_duration_tokens
                    if dense_duration_tokens
                    else None
                ),
                "preprocess_seconds": pre_seconds,
                "generation_seconds": durations,
                "generation_median_seconds": statistics.median(durations),
                "peak_cuda_gib": peak_gib,
                "gold": int(question["answer_id"]),
                "prediction": prediction,
                "correct": prediction == int(question["answer_id"]),
                "response": texts[0],
            }
            output.put(("sample", record))
    output.put(("done", {"worker": worker_index}))


def summarize(config: dict, records: list[dict], started: float) -> dict:
    by_method = {}
    for method in config["methods"]:
        name = method["name"]
        rows = [row for row in records if row["method"] == name]
        correct = sum(bool(row["correct"]) for row in rows)
        by_method[name] = {
            "samples": len(rows),
            "accuracy": correct / len(rows) if rows else None,
            "correct": correct,
            "retained_visual_tokens_median": statistics.median(
                row["retained_visual_tokens"] for row in rows
            ),
            "token_reduction_fraction_median": statistics.median(
                row["token_reduction_fraction"] for row in rows
            ),
            "preprocess_seconds_median": statistics.median(
                row["preprocess_seconds"] for row in rows
            ),
            "generation_seconds_median": statistics.median(
                row["generation_median_seconds"] for row in rows
            ),
            "end_to_end_seconds_median": statistics.median(
                row["preprocess_seconds"] + row["generation_median_seconds"]
                for row in rows
            ),
            "peak_cuda_gib_max": max(row["peak_cuda_gib"] for row in rows),
        }
    return {
        "schema_version": 1,
        "experiment": config["experiment"],
        "checkpoint": config["checkpoint"],
        "dataset": config["public_data"]["name"],
        "dataset_license": config["public_data"]["license"],
        "video_ids": config["video_ids"],
        "questions": len(records),
        "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
        "allocated_gpu_count": min(4, len(config["methods"]) * len(config["video_ids"])),
        "elapsed_seconds": time.perf_counter() - started,
        "methods": by_method,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    config = json.loads(Path(args.config).read_text())
    random.seed(int(config["seed"]))

    data_root, annotations = prepare_data(config)
    for video_id in config["video_ids"]:
        if video_id not in annotations:
            raise KeyError(f"missing annotations for {video_id}")

    from huggingface_hub import snapshot_download
    import torch

    checkpoint_path = snapshot_download(config["checkpoint"])
    config["_checkpoint_path"] = checkpoint_path
    gpu_count = torch.cuda.device_count()
    tasks = [
        {"method": method, "video_id": video_id}
        for method in config["methods"]
        for video_id in config["video_ids"]
    ]
    worker_count = min(gpu_count, 4, len(tasks))
    if worker_count < 1:
        raise RuntimeError("GPU-capable reproduction found no CUDA device")
    shards = [tasks[i::worker_count] for i in range(worker_count)]
    emit(
        "ORX_PROTOCOL_JSON",
        {
            key: value
            for key, value in config.items()
            if not key.startswith("_")
        }
        | {"worker_count": worker_count},
    )

    context = mp.get_context("spawn")
    output = context.Queue()
    processes = [
        context.Process(
            target=worker,
            args=(i, shards[i], config, annotations, str(data_root), output),
        )
        for i in range(worker_count)
    ]
    for process in processes:
        process.start()
    records = []
    completed = 0
    while completed < worker_count:
        try:
            kind, payload = output.get(timeout=30)
        except Empty:
            failed = [
                (process.pid, process.exitcode)
                for process in processes
                if process.exitcode not in (None, 0)
            ]
            if failed:
                raise RuntimeError(f"worker failure(s) before completion: {failed}")
            continue
        if kind == "sample":
            records.append(payload)
            emit("ORX_SAMPLE_JSON", payload)
        elif kind == "done":
            completed += 1
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"worker pid={process.pid} exited with {process.exitcode}"
            )
    result = summarize(config, records, started)
    emit("ORX_RESULT_JSON", result)


if __name__ == "__main__":
    main()
