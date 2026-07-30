#!/usr/bin/env python
"""Token and latency control on deterministic prefixes of a licensed open movie."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from queue import Empty

from run_reproduction import (
    emit,
    method_inputs,
    source_frame_count,
    timed_generate,
    visual_token_count,
)


def prepare_clips(config: dict) -> dict[str, Path]:
    root = Path(".cache/medium_control")
    root.mkdir(parents=True, exist_ok=True)
    source = root / "BigBuckBunny_320x180.mp4"
    archive = root / "BigBuckBunny_320x180.mp4.zip"
    if not source.exists():
        if not archive.exists():
            temp = archive.with_suffix(".part")
            request = urllib.request.Request(
                config["source"]["url"],
                headers={"User-Agent": "Mage-VL-reproduction/1.0"},
            )
            with urllib.request.urlopen(request) as response, temp.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            temp.replace(archive)
        subprocess.run(
            ["bsdtar", "-xf", str(archive), "-C", str(root)],
            check=True,
        )
    clips = {}
    for spec in config["clips"]:
        output = root / f"{spec['name']}.mp4"
        if not output.exists():
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", str(spec["start_seconds"]),
                    "-t", str(spec["duration_seconds"]),
                    "-i", str(source),
                    "-c", "copy",
                    str(output),
                ],
                check=True,
            )
        clips[spec["name"]] = output
    return clips


def prompt_for(processor) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {
                    "type": "text",
                    "text": (
                        "Describe the main visible events in chronological order. "
                        "Be concise and do not use outside knowledge."
                    ),
                },
            ],
        }
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def worker(index: int, tasks: list[dict], config: dict, clips: dict, queue) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    device = f"cuda:{index}"
    torch.cuda.set_device(index)
    processor = AutoProcessor.from_pretrained(
        config["_checkpoint_path"], trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["_checkpoint_path"],
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()
    merge = int(getattr(processor, "spatial_merge_size", 2))
    prompt = prompt_for(processor)
    for task in tasks:
        method = task["method"]
        clip = Path(clips[task["clip_name"]])
        local = dict(config, _device=device)
        cold_start = time.perf_counter()
        first_inputs = method_inputs(processor, prompt, clip, method, local)
        cold_seconds = time.perf_counter() - cold_start
        warm_start = time.perf_counter()
        inputs = method_inputs(processor, prompt, clip, method, local)
        warm_seconds = time.perf_counter() - warm_start
        durations, texts, peak_gib = timed_generate(
            model,
            processor,
            inputs,
            device,
            int(config["timing_repeats"]),
            int(config["max_new_tokens"]),
        )
        queue.put(
            (
                "sample",
                {
                    "experiment": config["experiment"],
                    "clip": task["clip_name"],
                    "clip_seconds": task["clip_seconds"],
                    "method": method["name"],
                    "backend": method["backend"],
                    "budget": method["budget"],
                    "source_video_frames": source_frame_count(clip),
                    "retained_visual_tokens": visual_token_count(inputs, merge),
                    "canvas_count": int(inputs["image_grid_thw"].shape[0]),
                    "cold_preprocess_seconds": cold_seconds,
                    "warm_preprocess_seconds": warm_seconds,
                    "generation_seconds": durations,
                    "generation_median_seconds": statistics.median(durations),
                    "peak_cuda_gib": peak_gib,
                    "response": texts[0],
                },
            )
        )
    queue.put(("done", {"worker": index}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    config = json.loads(Path(args.config).read_text())
    clips = prepare_clips(config)

    from huggingface_hub import snapshot_download
    import torch

    config["_checkpoint_path"] = snapshot_download(config["checkpoint"])
    tasks = [
        {
            "method": method,
            "clip_name": clip["name"],
            "clip_seconds": clip["duration_seconds"],
        }
        for method in config["methods"]
        for clip in config["clips"]
    ]
    workers = min(4, torch.cuda.device_count(), len(tasks))
    emit(
        "ORX_PROTOCOL_JSON",
        {k: v for k, v in config.items() if not k.startswith("_")}
        | {"worker_count": workers},
    )
    context = mp.get_context("spawn")
    queue = context.Queue()
    shards = [tasks[i::workers] for i in range(workers)]
    processes = [
        context.Process(
            target=worker,
            args=(i, shards[i], config, {k: str(v) for k, v in clips.items()}, queue),
        )
        for i in range(workers)
    ]
    for process in processes:
        process.start()
    rows = []
    finished = 0
    while finished < workers:
        try:
            kind, payload = queue.get(timeout=30)
        except Empty:
            failed = [
                (p.pid, p.exitcode)
                for p in processes
                if p.exitcode not in (None, 0)
            ]
            if failed:
                raise RuntimeError(f"worker failure(s): {failed}")
            continue
        if kind == "sample":
            rows.append(payload)
            emit("ORX_SAMPLE_JSON", payload)
        else:
            finished += 1
    for process in processes:
        process.join()
        if process.exitcode:
            raise RuntimeError(f"worker {process.pid} exited {process.exitcode}")
    emit(
        "ORX_RESULT_JSON",
        {
            "schema_version": 1,
            "experiment": config["experiment"],
            "source": config["source"],
            "diagnostic_only": True,
            "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
            "allocated_gpu_count": workers,
            "elapsed_seconds": time.perf_counter() - started,
            "measurements": rows,
        },
    )


if __name__ == "__main__":
    main()
