import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import marimo as mo

    return math, mo


@app.cell
def _():
    results = {
        "budget": [4, 8, 16, 32],
        "codec_accuracy": [72.22, 72.22, 69.44, 75.00],
        "codec_tokens": [576, 1152, 2304, 4608],
        "codec_e2e": [0.1993, 0.3295, 0.6363, 1.2080],
        "uniform_accuracy": [77.78, 80.56, 72.22, 72.22],
        "uniform_tokens": [6032, 12064, 24128, 48256],
        "uniform_e2e": [1.038, 2.2272, 4.395, 7.722],
        "accuracy_delta_ci95": [
            [-13.89, 0.00],
            [-19.44, 0.00],
            [-13.89, 5.56],
            [-5.56, 13.89],
        ],
        "speedup": [5.21, 6.76, 6.91, 6.39],
        "matched_duration": {
            "codec_accuracy": 72.22,
            "uniform_accuracy": 75.00,
            "codec_tokens": 1152,
            "uniform_tokens": 96512,
            "speedup": 71.68,
        },
        "token_matched": {
            "codec_accuracy": 69.44,
            "uniform_accuracy": 80.56,
            "codec_tokens": 12672,
            "uniform_tokens": 12064,
            "codec_e2e": 5.466,
            "uniform_e2e": 2.255,
        },
        "alternate_codec": {
            "hevc_correct": 26,
            "dcvc_correct": 26,
            "questions": 36,
            "median_jaccard": 0.1725,
        },
    }
    return (results,)


@app.cell
def _(math):
    def evidence_svg(results):
        points = []
        for i, budget in enumerate(results["budget"]):
            points.append((results["codec_tokens"][i], results["codec_accuracy"][i], "#146C94", f"tc{budget}"))
            points.append((results["uniform_tokens"][i], results["uniform_accuracy"][i], "#D95F02", f"u{budget}"))
        tm = results["token_matched"]
        points.append((tm["codec_tokens"], tm["codec_accuracy"], "#146C94", "tc84*"))
        xmin, xmax, ymin, ymax = 500, 100000, 66, 83
        xp = lambda x: 65 + (math.log10(x) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin)) * 700
        yp = lambda y: 370 - (y - ymin) / (ymax - ymin) * 285
        out = [
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 430" width="100%">',
            '<style>text{font-family:Inter,Arial,sans-serif;fill:#17202A}.t{font-size:13px}.h{font-size:21px;font-weight:700}.g{stroke:#D5D8DC}</style>',
            '<rect width="820" height="430" fill="#FCFCFA"/>',
            '<text x="48" y="32" class="h">Observed accuracy versus measured visual tokens</text>',
        ]
        for y in [68, 72, 76, 80]:
            out += [f'<line x1="65" y1="{yp(y)}" x2="765" y2="{yp(y)}" class="g"/>',
                    f'<text x="55" y="{yp(y)+4}" text-anchor="end" class="t">{y}%</text>']
        for x, y, color, label in points:
            out += [f'<circle cx="{xp(x)}" cy="{yp(y)}" r="7" fill="{color}" stroke="white" stroke-width="2"/>',
                    f'<text x="{xp(x)+9}" y="{yp(y)-8}" class="t">{label}</text>']
        out += [
            '<text x="415" y="414" text-anchor="middle" class="t">retained visual tokens (log scale)</text>',
            '<circle cx="575" cy="28" r="6" fill="#146C94"/><text x="586" y="33" class="t">codec</text>',
            '<circle cx="670" cy="28" r="6" fill="#D95F02"/><text x="681" y="33" class="t">uniform</text>',
            '</svg>',
        ]
        return "".join(out)

    return (evidence_svg,)


@app.cell
def _(evidence_svg, mo, results):
    mo.vstack(
        [
            mo.md(
                """
    # Mage-VL codec-native reproduction

    Video-language models normally inspect many nearly redundant image patches. Mage-VL asks the video codec which regions changed and spends model tokens there; this bounded reproduction tests whether that saves tokens and wall time without losing answers.

    **Verdict — partially reproduced.** At matched 64-frame duration, HEVC-guided selection removed **98.8%** of measured visual tokens and was **71.7× faster**, with a small and statistically uncertain accuracy difference. An approximately token-matched stress test did not preserve accuracy or latency.
    """
            ),
            mo.Html(evidence_svg(results)),
            mo.md(
                "*Blue is codec-guided and orange is uniform sampling. Farther left is cheaper; higher is more accurate. `tc84*` is the approximately token-matched stress test.*"
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(mo, results):
    rows = []
    for i, budget in enumerate(results["budget"]):
        rows.append(
            {
                "nominal N": budget,
                "codec accuracy": f"{results['codec_accuracy'][i]:.2f}%",
                "uniform accuracy": f"{results['uniform_accuracy'][i]:.2f}%",
                "codec tokens": f"{results['codec_tokens'][i]:,}",
                "uniform tokens": f"{results['uniform_tokens'][i]:,}",
                "speedup": f"{results['speedup'][i]:.2f}×",
                "paired Δ 95% CI (pp)": str(results["accuracy_delta_ci95"][i]),
            }
        )
    mo.vstack(
        [
            mo.md(
                """
    ## Budget sweep

    The 36-question public Perception Test slice is deliberately small, so the paired accuracy intervals are wide. The stable result is the systems trend: sparse inference is much cheaper at every nominal budget.
    """
            ),
            mo.ui.table(rows, selection=None),
        ]
    )
    return


@app.cell
def _(mo, results):
    md = results["matched_duration"]
    tm = results["token_matched"]
    ac = results["alternate_codec"]
    mo.md(
        f"""
    ## Three tests, three different questions

    1. **Matched source duration.** `tc8` packs salient patches from the same 64-frame horizon used by dense uniform processing: {md['codec_tokens']:,} versus {md['uniform_tokens']:,} tokens, {md['codec_accuracy']:.2f}% versus {md['uniform_accuracy']:.2f}% accuracy, and {md['speedup']:.2f}× speedup.
    2. **Approximately matched measured tokens.** `tc84` used {tm['codec_tokens']:,} tokens versus uniform8's {tm['uniform_tokens']:,}, but reached {tm['codec_accuracy']:.2f}% versus {tm['uniform_accuracy']:.2f}% and took {tm['codec_e2e']:.3f}s versus {tm['uniform_e2e']:.3f}s. This setup did not show the paper's fixed-budget advantage.
    3. **Alternate codec.** HEVC and DCVC-RT both answered {ac['hevc_correct']}/{ac['questions']} correctly without retraining, even though their median selected-patch Jaccard overlap was only {100 * ac['median_jaccard']:.1f}%.

    These are not interchangeable claims. The first supports the headline token-reduction mechanism, the second is a negative stress test, and the third supports downstream robustness rather than identical rankings.
    """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Reproduction design and limits

    - **Data:** eight real 25–35 second videos and 36 multiple-choice questions from `lmms-lab/PerceptionTest`'s licensed sample split.
    - **Model:** released `microsoft/Mage-VL` checkpoint; HEVC and bundled DCVC-RT codec paths; greedy single-sample inference.
    - **Metrics:** retained visual embeddings, preprocessing, generation, warm preprocessing-inclusive latency, peak allocated GPU memory, and option correctness.
    - **Uncertainty:** 10,000 paired question-level bootstrap resamples, seed `260724904`.
    - **Compute:** Kubernetes; NVIDIA RTX PRO 6000 Blackwell; four GPUs allocated per experiment; 16 GPUs peak concurrent; 0.40 hours campaign wall time.

    This is a bounded slice, not a replication of the paper's headline NExT-QA table. It uses Mage-VL's own uniform processor, not Qwen3-VL, and question-level resampling. The expensive evidence is embedded above; opening this notebook does not rerun model inference.
    """)
    return


if __name__ == "__main__":
    app.run()
