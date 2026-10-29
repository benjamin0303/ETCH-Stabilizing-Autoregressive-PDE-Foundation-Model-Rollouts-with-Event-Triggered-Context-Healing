"""Stride-20 DPOT rollouts and the two matched baselines."""
import time

import numpy as np
import torch

from etch.reconstruct import (
    append_field_metrics, dpot_blocks, reconstruct_full_context, tap_relative_l2,
)

TIMING_COLUMNS = ["discrepancy", "trigger", "reconstruction", "context_update", "dpot", "total"]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize(arrays, frame_interval_ms=5.):
    summary = {"n_frames": len(arrays["frame_index"])}
    summary["mean_errors"] = {
        key: float(value.mean()) for key, value in arrays.items() if key.endswith("rel_l2")
    }
    complete = arrays["complete"]
    summary["n_complete_windows"] = int(complete.sum())
    if "correction_applied" in arrays:
        count = int(arrays["correction_applied"].sum())
        summary.update(corrections=count, correction_fraction=count / max(1, int(complete.sum())))
    if "timings_ms" in arrays:
        timings = arrays["timings_ms"]
        total = timings[:, -1] if timings.ndim == 2 else timings
        measured = total[complete]
        if len(measured):
            summary.update(
                mean_cycle_ms=float(measured.mean()),
                median_cycle_ms=float(np.median(measured)),
                mean_csr=float(measured.mean() / (20 * frame_interval_ms)),
            )
    return summary


@torch.inference_mode()
def run_etch(velocity, pressure, dpot, normalizer, truth, taps, pixels, reynolds,
             seed, threshold=.05, frame_interval_ms=5., save_fields=False):
    device = velocity.device
    stop_frame = len(truth)
    context, _ = reconstruct_full_context(velocity, pressure, taps[:40], 20, reynolds, pixels, seed)
    for _ in range(5):
        dpot_blocks(context[None], dpot, normalizer, device)
    prediction, _ = dpot_blocks(context[None], dpot, normalizer, device)
    issued_errors = {key: [] for key in ("full", "uv", "p")}
    corrected_errors = {key: [] for key in ("full", "uv", "p")}
    starts, stops, scores, corrected, timings, fields = [], [], [], [], [], []
    for block, start in enumerate(range(40, stop_frame, 20)):
        stop = min(start + 20, stop_frame)
        valid = stop - start
        issued = prediction[0, :valid]
        synchronize(device)
        t0 = time.perf_counter_ns()
        score = tap_relative_l2(issued, taps[start:stop], pixels)
        t1 = time.perf_counter_ns()
        apply = bool(score > threshold and valid == 20)
        t2 = time.perf_counter_ns()
        if apply:
            healed, _ = reconstruct_full_context(
                velocity, pressure, taps[:stop], stop - 20, reynolds, pixels,
                seed + (block + 1) * 100,
            )
        t3 = time.perf_counter_ns()
        if apply:
            context = healed.clone()
            field = healed
        else:
            context = torch.cat((context[valid:], issued))
            field = issued
        contexts = context[None]
        t4 = time.perf_counter_ns()
        if stop < stop_frame:
            prediction, _ = dpot_blocks(contexts, dpot, normalizer, device)
        synchronize(device)
        t5 = time.perf_counter_ns()
        timings.append([(t1-t0)/1e6, (t2-t1)/1e6, (t3-t2)/1e6,
                        (t4-t3)/1e6, (t5-t4)/1e6, (t5-t0)/1e6])
        starts.append(start); stops.append(stop); scores.append(score); corrected.append(apply)
        append_field_metrics(issued_errors, issued, truth[start:stop])
        append_field_metrics(corrected_errors, field, truth[start:stop])
        if save_fields:
            fields.append(field.numpy())
    starts, stops = np.asarray(starts), np.asarray(stops)
    arrays = dict(
        frame_index=np.arange(40, stop_frame), block_start=starts, block_stop=stops,
        complete=stops-starts == 20, pressure_tap_discrepancy=np.asarray(scores),
        correction_applied=np.asarray(corrected), timings_ms=np.asarray(timings),
    )
    for prefix, errors in (("issued", issued_errors), ("post_replacement", corrected_errors)):
        arrays.update({f"{prefix}_{key}_rel_l2": np.concatenate(parts) for key, parts in errors.items()})
    backlog, delay = [], 0.
    for elapsed, length in zip(arrays["timings_ms"][:, -1], stops-starts):
        delay = max(0., delay + elapsed - length * frame_interval_ms)
        backlog.append(delay)
    arrays["backlog_ms"] = np.asarray(backlog)
    if save_fields:
        arrays["fields"] = np.concatenate(fields)
    return arrays


@torch.inference_mode()
def run_hdit(velocity, pressure, truth, taps, pixels, reynolds, seed, save_fields=False):
    metrics = {key: [] for key in ("full", "uv", "p")}
    timings, complete, fields = [], [], []
    for block, start in enumerate(range(40, len(truth), 20)):
        stop = min(start + 20, len(truth))
        synchronize(velocity.device)
        before = time.perf_counter_ns()
        field, _ = reconstruct_full_context(
            velocity, pressure, taps[:stop], stop-20, reynolds, pixels,
            seed + (block + 1) * 100,
        )
        synchronize(velocity.device)
        timings.append((time.perf_counter_ns() - before) / 1e6)
        valid = stop-start
        append_field_metrics(metrics, field[-valid:], truth[start:stop])
        complete.append(valid == 20)
        if save_fields:
            fields.append(field[-valid:].numpy())
    arrays = {key+"_rel_l2": np.concatenate(parts) for key, parts in metrics.items()}
    arrays.update(frame_index=np.arange(40, len(truth)), complete=np.asarray(complete),
                  timings_ms=np.asarray(timings))
    if save_fields:
        arrays["fields"] = np.concatenate(fields)
    return arrays


@torch.inference_mode()
def run_dpot(context, dpot, normalizer, truth, device, save_fields=False):
    metrics = {key: [] for key in ("full", "uv", "p")}
    complete, fields = [], []
    for start in range(40, len(truth), 20):
        prediction, _ = dpot_blocks(context[None], dpot, normalizer, device)
        valid = min(20, len(truth)-start)
        append_field_metrics(metrics, prediction[0, :valid], truth[start:start+valid])
        context = prediction[0]
        complete.append(valid == 20)
        if save_fields:
            fields.append(prediction[0, :valid].numpy())
    arrays = {key+"_rel_l2": np.concatenate(parts) for key, parts in metrics.items()}
    arrays.update(frame_index=np.arange(40, len(truth)), complete=np.asarray(complete))
    if save_fields:
        arrays["fields"] = np.concatenate(fields)
    return arrays
