"""Stage 0 Go/No-Go: Singular value spectrum of MoE dispatch activations.

Hooks the input of each selected MoE sparse block in Qwen3-30B-A3B, runs a small
calibration batch, then performs SVD on the activation matrix X in R^{N x d}.

Outputs (under eval_results/compression_lowrank_stage0/):
  - activations_layer{L}.pt    : the captured float tensor [N, d]
  - svd_summary.json           : per-layer singular values, cumulative variance,
                                 fractions at l = d/8, d/4, d/2, and Go/No-Go verdict
  - svd_curve.png              : cumulative explained variance vs l/d, one line per layer
  - report.md                  : human-readable summary

Run:
  /home/lzy/miniconda3/envs/atom/bin/python eval/compression/stage0_low_rank_svd.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

MODEL_PATH = "/home/lzy/models/Qwen3-30B-A3B"
OUT_DIR = Path("eval_results/compression_lowrank_stage0")


# A small, domain-diverse calibration set. Keeping prompts varied so the
# spectrum is not dominated by one genre.
CALIBRATION_PROMPTS = [
    # General-knowledge / encyclopedia (5)
    "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic cells. It is responsible for generating most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy. Mitochondria are commonly between 0.75 and 3 micrometers in diameter but vary considerably in size and structure.",
    "The French Revolution was a period of radical political and societal change in France that began with the Estates General of 1789 and ended with the formation of the French Consulate in November 1799. Many of its ideas are considered fundamental principles of liberal democracy, while its values and institutions remain central to French political discourse.",
    "Black holes are regions of spacetime where gravity is so strong that nothing, not even light or other electromagnetic waves, has enough energy to escape. Albert Einstein's theory of general relativity predicts that a sufficiently compact mass can deform spacetime to form a black hole.",
    "The Renaissance was a period in European history marking the transition from the Middle Ages to modernity and covering the 15th and 16th centuries. In addition to the standard periodization, proponents of a long Renaissance start it in the 14th century and end it in the 17th century.",
    "Photosynthesis is a biological process used by many cellular organisms to convert light energy into chemical energy, which is stored in organic compounds that can later be metabolized through cellular respiration to fuel the organism's activities.",
    # Code / Python (6)
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n\nprint(quicksort([3, 6, 8, 10, 1, 2, 1]))",
    "import torch\nimport torch.nn as nn\n\nclass MLP(nn.Module):\n    def __init__(self, dim, hidden):\n        super().__init__()\n        self.fc1 = nn.Linear(dim, hidden)\n        self.act = nn.SiLU()\n        self.fc2 = nn.Linear(hidden, dim)\n\n    def forward(self, x):\n        return self.fc2(self.act(self.fc1(x)))\n\nmodel = MLP(256, 1024).cuda()",
    "// JavaScript: debounce helper\nfunction debounce(fn, wait) {\n  let timer;\n  return function (...args) {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn.apply(this, args), wait);\n  };\n}\n\nconst onResize = debounce(() => console.log('resized'), 200);\nwindow.addEventListener('resize', onResize);",
    "// Rust: error handling with Result\nuse std::fs::File;\nuse std::io::{self, Read};\n\nfn read_file(path: &str) -> io::Result<String> {\n    let mut f = File::open(path)?;\n    let mut s = String::new();\n    f.read_to_string(&mut s)?;\n    Ok(s)\n}\n\nfn main() {\n    match read_file(\"/etc/hostname\") {\n        Ok(s) => println!(\"{}\", s),\n        Err(e) => eprintln!(\"err: {}\", e),\n    }\n}",
    "SELECT u.id, u.name, COUNT(o.id) AS order_count, SUM(o.total) AS revenue\nFROM users u\nLEFT JOIN orders o ON o.user_id = u.id AND o.created_at >= '2025-01-01'\nWHERE u.country = 'JP'\nGROUP BY u.id, u.name\nHAVING COUNT(o.id) > 0\nORDER BY revenue DESC\nLIMIT 50;",
    "# Go: simple HTTP server\npackage main\n\nimport (\n    \"fmt\"\n    \"log\"\n    \"net/http\"\n)\n\nfunc handler(w http.ResponseWriter, r *http.Request) {\n    fmt.Fprintf(w, \"hello %s\\n\", r.URL.Path[1:])\n}\n\nfunc main() {\n    http.HandleFunc(\"/\", handler)\n    log.Fatal(http.ListenAndServe(\":8080\", nil))\n}",
    # Math / reasoning (5)
    "A train leaves station A at 8:00 traveling east at 60 km/h. A second train leaves station B (300 km east of A) at 9:00 traveling west at 90 km/h. At what time do the trains meet, and how far from station A is the meeting point? Let t be the hours after 8:00 when they meet. The first train has covered 60 t km. The second train starts an hour later at 9:00, so by time t it has been traveling t - 1 hours and covered 90 (t - 1) km. They meet when 60 t + 90 (t - 1) = 300.",
    "Prove that the square root of 2 is irrational. Suppose for contradiction that sqrt(2) = p/q where p and q are coprime integers. Then 2 q^2 = p^2, so p^2 is even, hence p is even. Write p = 2 k. Then 2 q^2 = 4 k^2, so q^2 = 2 k^2, hence q is also even. This contradicts the assumption that p and q are coprime.",
    "Let f(x) = x^3 - 3 x + 1. Find all real roots. f'(x) = 3 x^2 - 3 = 3 (x - 1)(x + 1). Critical points at x = -1 (local max, f(-1) = 3) and x = 1 (local min, f(1) = -1). Since the local min is negative and the local max is positive, f has three distinct real roots.",
    "Compute the integral of x e^x from 0 to 1. Using integration by parts with u = x, dv = e^x dx, we get du = dx, v = e^x. So integral = x e^x - integral e^x dx = x e^x - e^x. Evaluated from 0 to 1: (e - e) - (0 - 1) = 1.",
    "Solve the system: 2 x + 3 y = 13, 4 x - y = 5. From the second equation y = 4 x - 5. Substituting: 2 x + 3 (4 x - 5) = 13, so 14 x - 15 = 13, hence x = 2 and y = 3.",
    # Dialog / chat-style (5)
    "User: Could you help me draft a polite email declining a meeting invitation? Assistant: Of course! Here is a polite template: Dear [Name], Thank you very much for the kind invitation to the meeting on [date]. Unfortunately, I have a prior commitment at that time and will not be able to attend. I would love to stay informed about the discussion, so if a summary or notes are available afterwards I would greatly appreciate seeing them. Best regards, [Your name]",
    "User: Recommend three classic novels for a teenager. Assistant: Here are three highly recommended classics. To Kill a Mockingbird by Harper Lee, a coming-of-age story about justice and prejudice in the American South. The Hobbit by J.R.R. Tolkien, an accessible introduction to fantasy adventure. The Catcher in the Rye by J.D. Salinger, a witty and emotional exploration of adolescence.",
    "User: How does HTTPS encrypt traffic? Assistant: HTTPS uses Transport Layer Security on top of TCP. The handshake begins with the client sending a list of supported ciphers and a random nonce. The server responds with its certificate and selected cipher. They derive shared symmetric keys via an asymmetric key exchange such as ECDHE, then switch to symmetric encryption like AES-GCM for the actual data.",
    "User: What is your favorite book? Assistant: As an AI I do not have personal preferences, but a book I often recommend is Thinking Fast and Slow by Daniel Kahneman, because it offers a clear framework for understanding human reasoning that applies to many practical situations.",
    "User: Translate this sentence to French: I would like to order a coffee and a croissant. Assistant: Je voudrais commander un cafe et un croissant.",
    # Long-form / narrative (5)
    "The cabin sat at the edge of the lake, half hidden by pines. Inside, an old wood stove ticked as it cooled, and on the table a half-finished letter waited for its writer to come back from the dock. Outside, the loons called once, twice, then went quiet, and the long Northern summer evening began.",
    "By the time the rain reached the city, it was already a different rain than the one that had started over the mountains. It had picked up pine needles and dust and the cold smell of stone, and when it finally tapped at the windows it sounded a little like an old friend asking to be let in.",
    "The conference room had that particular kind of fluorescent quiet, where the hum of the lights becomes the only honest thing in the room. Jana clicked to the last slide of her deck and waited for the question she knew was coming, and was almost relieved when it came.",
    "Three things were now certain: the package had arrived, no one in the apartment building remembered ordering it, and whatever was inside was warm. Mrs. Halvorsen, who lived in 4B and was rumored to know everyone's business, claimed not to know whose name was on the label, which was the most suspicious detail of all.",
    "The mountain refused to be hurried. That was the lesson the new climbers always learned first, usually somewhere around the second hour of the approach hike, when the lake below stopped getting closer and the trees started to thin and the air started to bite at the back of the throat in a way that felt almost personal.",
    # News / financial (4)
    "Markets opened lower across Asia on Monday after weaker-than-expected manufacturing data out of China and a renewed dispute over export controls on advanced semiconductors. The Nikkei 225 closed down 1.4 percent, while the Hang Seng index slipped 1.8 percent. Bond yields in the region were broadly steady as traders awaited the U.S. inflation print due later in the week.",
    "The central bank held its policy rate at 4.25 percent for a third consecutive meeting, citing persistent services inflation and a labor market that has cooled only modestly. In its accompanying statement, the committee said additional tightening could not be ruled out if upside risks to inflation materialize over the coming quarters.",
    "Annual revenue at the cloud unit grew 32 percent year over year, the company reported, driven by strong demand for inference workloads and a multi-year contract win in the public sector. Operating margin expanded by roughly two percentage points despite continued investment in data-center capacity and custom accelerator design.",
    "Regulators in the European Union opened a formal investigation into the proposed acquisition, focusing on potential effects on competition in the market for industrial control software. The companies said they were confident the deal could be cleared with limited remedies and reiterated their target of closing by mid-2026.",
    # Science explanation (5)
    "The greenhouse effect refers to the warming of a planet's lower atmosphere caused by certain gases absorbing and re-emitting infrared radiation. On Earth, the principal contributors are water vapor, carbon dioxide, methane, nitrous oxide, and ozone. Without any greenhouse effect the surface would average roughly minus 18 degrees Celsius rather than the observed plus 14 degrees Celsius.",
    "An action potential is a rapid change in the electrical potential across the membrane of a neuron, typically triggered when the membrane depolarizes past a threshold of about minus 55 millivolts. Voltage-gated sodium channels open, sodium rushes in, and the membrane briefly reverses polarity before potassium channels open and restore the resting potential.",
    "Quantum entanglement describes a non-classical correlation between two or more quantum systems, such that the state of each cannot be described independently. Measurements on entangled systems are correlated in ways that cannot be reproduced by any local hidden-variable theory, as Bell's theorem demonstrates.",
    "The CRISPR-Cas9 system is a defense mechanism originally found in bacteria, now widely used as a precise genome-editing tool. A guide RNA directs the Cas9 nuclease to a target sequence, where it introduces a double-stranded break that the cell then repairs, often imperfectly, allowing researchers to insert, delete, or modify specific genes.",
    "Convection in the Earth's mantle drives plate tectonics. Hot material rises near mid-ocean ridges, spreads outward across the seafloor, cools, and eventually sinks back into the mantle at subduction zones. The resulting motion of lithospheric plates produces most of the world's earthquakes and a great deal of its volcanism.",
    # Conversational instruction (5)
    "Please summarize the following passage in two sentences. The passage describes a hiking trip during which the narrator first encounters the unique flora of a high-altitude meadow, including alpine forget-me-nots and dwarf willow, while reflecting on how long it took to reach this particular ridge after years of putting the trip off.",
    "Write a short product description for a portable espresso maker designed for travelers. It should fit in a backpack, weigh less than 500 grams, work without electricity, and produce real crema. Mention durability, ease of cleaning, and a one-year warranty.",
    "Explain to a beginner what a database index is and when one should be added. An index is a separate data structure that lets the database engine locate rows that match a condition without scanning the entire table. They speed up reads but slow down writes and consume storage, so they are usually added when a query is too slow and uses a specific column in its WHERE or JOIN clause.",
    "Compare and contrast supervised learning, unsupervised learning, and reinforcement learning in plain language. Supervised learning predicts labels from labeled examples. Unsupervised learning finds structure in unlabeled data, such as clusters or low-dimensional embeddings. Reinforcement learning learns a policy by acting in an environment and receiving rewards or penalties for those actions.",
    "Write a step-by-step recipe for a simple weeknight pasta dish that takes under thirty minutes and uses ingredients commonly found in a typical kitchen. Begin by boiling salted water, then while the pasta cooks, prepare a garlic and olive oil sauce in a separate pan, finishing the dish by tossing the drained pasta in the pan with a generous handful of grated parmesan.",
    # Multilingual snippets (4)
    "El sistema solar es el sistema planetario en el que se encuentran la Tierra y otros objetos astronomicos que giran directa o indirectamente en una orbita alrededor de una unica estrella conocida como el Sol. Comprende ocho planetas y cinco planetas enanos oficialmente reconocidos.",
    "Le langage Python est un langage de programmation interprete, multiparadigme et multiplateformes. Il favorise la programmation imperative structuree, fonctionnelle et orientee objet. Il dispose d'un typage dynamique fort, d'une gestion automatique de la memoire par ramasse-miettes et d'un systeme de gestion d'exceptions.",
    "Die Quantenmechanik ist eine physikalische Theorie zur Beschreibung der Materie und der Wechselwirkungen auf atomarer und subatomarer Ebene. Im Gegensatz zur klassischen Physik beschreibt sie Zustande nicht als Punkte im Phasenraum, sondern als Vektoren in einem Hilbert-Raum.",
    "Tokyo, officially the Tokyo Metropolis, is the capital and most populous city of Japan. Originally a fishing village named Edo, the city became a prominent political center in 1603 when it became the seat of the Tokugawa shogunate, transforming into one of the world's most populous urban centers.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", default=MODEL_PATH)
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[2, 12, 24, 36, 46],
        help="MoE layer indices to hook (0-indexed)",
    )
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-prompts", type=int, default=64)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-mem-gib", type=int, default=22, help="per-GPU cap for device_map auto")
    return p.parse_args()


def setup_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    dtype = setup_dtype(args.dtype)

    print(f"[stage0] loading {args.model_path}  dtype={args.dtype}", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_mem = {i: f"{args.gpu_mem_gib}GiB" for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        max_memory=max_mem,
        trust_remote_code=True,
    )
    model.eval()
    cfg = model.config
    hidden_size = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    print(
        f"[stage0] loaded: hidden_size={hidden_size}, num_hidden_layers={n_layers}, "
        f"num_experts={cfg.num_experts}, top_k={cfg.num_experts_per_tok}",
        flush=True,
    )

    layers_to_hook = [L for L in args.layers if 0 <= L < n_layers]
    print(f"[stage0] hooking MoE input on layers {layers_to_hook}", flush=True)

    # Buffers for captured activations and per-token gate decisions.
    activations: dict[int, list[torch.Tensor]] = {L: [] for L in layers_to_hook}
    # Also capture routing top-1 expert per token to enable a per-expert SVD as a bonus.
    routings: dict[int, list[torch.Tensor]] = {L: [] for L in layers_to_hook}
    attn_mask_for_run: torch.Tensor | None = None

    def make_mlp_pre_hook(layer_id: int):
        def hook(module, inputs):
            # inputs[0] shape [B, T, H] in HF qwen3_moe
            x = inputs[0].detach()
            activations[layer_id].append(x.to("cpu", dtype=torch.float32).reshape(-1, x.shape[-1]))
            return None
        return hook

    def make_gate_hook(layer_id: int):
        def hook(module, inputs, output):
            # output: gate logits [B*T, E] or [B, T, E]; argmax over E -> top1 expert
            logits = output.detach()
            top1 = logits.argmax(dim=-1).to("cpu").reshape(-1)
            routings[layer_id].append(top1)
            return None
        return hook

    handles = []
    for L in layers_to_hook:
        layer = model.model.layers[L]
        mlp = layer.mlp
        handles.append(mlp.register_forward_pre_hook(make_mlp_pre_hook(L)))
        # In HF Qwen3MoeSparseMoeBlock, the router is `mlp.gate` (nn.Linear).
        if hasattr(mlp, "gate"):
            handles.append(mlp.gate.register_forward_hook(make_gate_hook(L)))

    # Build batched calibration tensors.
    prompts = CALIBRATION_PROMPTS[: args.max_prompts]
    print(f"[stage0] tokenizing {len(prompts)} prompts (max_seq_len={args.max_seq_len})", flush=True)
    encoded = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=args.max_seq_len,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attn_mask = encoded["attention_mask"]
    # Move to the device of embed_tokens (whichever GPU device_map placed it on).
    embed_dev = next(model.model.embed_tokens.parameters()).device
    input_ids = input_ids.to(embed_dev)
    attn_mask = attn_mask.to(embed_dev)

    n_tokens_kept = int(attn_mask.sum().item())
    print(f"[stage0] {input_ids.shape[0]} prompts x {input_ids.shape[1]} positions = {input_ids.numel()} tokens, kept={n_tokens_kept}", flush=True)

    # Run in mini-batches to keep memory bounded.
    bsz = args.batch_size
    with torch.no_grad():
        for i in range(0, input_ids.shape[0], bsz):
            ids = input_ids[i : i + bsz]
            am = attn_mask[i : i + bsz]
            _ = model(input_ids=ids, attention_mask=am, use_cache=False)
            print(f"[stage0]   batch {i // bsz + 1}/{(input_ids.shape[0] + bsz - 1) // bsz} done", flush=True)

    for h in handles:
        h.remove()

    # Reshape and apply attention mask to drop padding tokens.
    am_flat = attn_mask.to("cpu").bool().reshape(-1)
    print(f"[stage0] keep mask: {int(am_flat.sum())} / {am_flat.numel()} tokens", flush=True)

    summary: dict = {
        "model_path": args.model_path,
        "hidden_size": hidden_size,
        "num_hidden_layers": n_layers,
        "num_experts": int(cfg.num_experts),
        "top_k": int(cfg.num_experts_per_tok),
        "n_calibration_prompts": len(prompts),
        "max_seq_len": int(args.max_seq_len),
        "n_kept_tokens": int(am_flat.sum().item()),
        "layers": {},
        "go_criteria": {
            "explained_variance_at_d_over_4": 0.90,
            "rule": "GO if explained variance at l = hidden_size/4 >= 0.90 in ALL probed layers",
        },
    }

    for L in layers_to_hook:
        chunks = activations[L]
        if not chunks:
            print(f"[stage0] layer {L}: no activations captured (skipping)", flush=True)
            continue
        X = torch.cat(chunks, dim=0)  # [B*T, H]
        if X.shape[0] != am_flat.shape[0]:
            print(f"[stage0]   layer {L}: shape mismatch X={X.shape[0]} vs mask={am_flat.shape[0]} -- using min", flush=True)
            n = min(X.shape[0], am_flat.shape[0])
            X = X[:n]
            mask_l = am_flat[:n]
        else:
            mask_l = am_flat
        X = X[mask_l]  # drop pads
        N, d = X.shape
        print(f"[stage0] layer {L}: activation matrix shape ({N}, {d})", flush=True)

        # Save raw for downstream re-analysis.
        torch.save({"X": X, "layer": L}, out_dir / f"activations_layer{L}.pt")

        # Center for "explained variance" semantics (PCA convention). Spectrum
        # of uncentered X is also reported as a sanity-check.
        X_centered = X - X.mean(dim=0, keepdim=True)

        # torch.linalg.svdvals returns singular values descending.
        sv_centered = torch.linalg.svdvals(X_centered).cpu().numpy()
        sv_raw = torch.linalg.svdvals(X).cpu().numpy()

        # Explained variance = sigma^2 / sum(sigma^2)
        import numpy as np
        ev_c = sv_centered ** 2
        cum_c = np.cumsum(ev_c) / ev_c.sum()
        ev_r = sv_raw ** 2
        cum_r = np.cumsum(ev_r) / ev_r.sum()

        # Sample at l = d/8, d/4, d/2.
        marks = {"d/8": d // 8, "d/4": d // 4, "d/2": d // 2}
        layer_info = {
            "shape": [int(N), int(d)],
            "singular_values_centered_first16": [float(x) for x in sv_centered[:16]],
            "singular_values_centered_last4": [float(x) for x in sv_centered[-4:]],
            "explained_variance_centered_at": {
                k: float(cum_c[min(v, len(cum_c)) - 1]) for k, v in marks.items()
            },
            "explained_variance_raw_at": {
                k: float(cum_r[min(v, len(cum_r)) - 1]) for k, v in marks.items()
            },
            # How many ell are needed to reach a few thresholds?
            "rank_at_explained_variance": {
                "0.90": int(np.searchsorted(cum_c, 0.90) + 1),
                "0.95": int(np.searchsorted(cum_c, 0.95) + 1),
                "0.99": int(np.searchsorted(cum_c, 0.99) + 1),
            },
        }
        summary["layers"][str(L)] = layer_info

        # Save full curve for plotting.
        np.save(out_dir / f"cum_var_centered_layer{L}.npy", cum_c)
        np.save(out_dir / f"cum_var_raw_layer{L}.npy", cum_r)
        np.save(out_dir / f"sv_centered_layer{L}.npy", sv_centered)

        # Top-1 expert assignment (uncomprehensive — informational only).
        if routings[L]:
            top1 = torch.cat(routings[L], dim=0)
            if top1.numel() == am_flat.numel():
                top1 = top1[am_flat]
            else:
                # gate might be applied to a different shape (e.g., flattened);
                # take last N entries that match mask length.
                top1 = top1[: am_flat.shape[0]][am_flat]
            uniq, cnt = torch.unique(top1, return_counts=True)
            layer_info["top1_unique_experts"] = int(uniq.numel())
            layer_info["top1_expert_count_top5"] = [
                [int(uniq[i]), int(cnt[i])]
                for i in torch.argsort(cnt, descending=True)[:5].tolist()
            ]

    # Apply Go/No-Go.
    all_ge_90 = True
    layer_ev_d4 = {}
    for L, info in summary["layers"].items():
        v = info["explained_variance_centered_at"]["d/4"]
        layer_ev_d4[L] = v
        if v < 0.90:
            all_ge_90 = False
    summary["go_criteria"]["per_layer_explained_variance_at_d_over_4"] = layer_ev_d4
    summary["go_criteria"]["verdict"] = "GO" if all_ge_90 else "NO-GO"

    with (out_dir / "svd_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[stage0] wrote {out_dir/'svd_summary.json'}", flush=True)

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        for L in layers_to_hook:
            p = out_dir / f"cum_var_centered_layer{L}.npy"
            if not p.exists():
                continue
            cum = np.load(p)
            d = summary["layers"][str(L)]["shape"][1]
            xs = np.arange(1, len(cum) + 1) / d
            ax.plot(xs, cum, label=f"layer {L}")
        ax.axvline(0.25, color="gray", linestyle="--", linewidth=0.8, label="l/d = 1/4")
        ax.axhline(0.90, color="red", linestyle="--", linewidth=0.8, label="0.90 threshold")
        ax.set_xlabel("l / d (kept rank fraction)")
        ax.set_ylabel("cumulative explained variance (centered)")
        ax.set_title(f"MoE dispatch activation spectrum — Qwen3-30B-A3B (d={hidden_size})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "svd_curve.png", dpi=140)
        print(f"[stage0] wrote {out_dir/'svd_curve.png'}", flush=True)
    except Exception as e:
        print(f"[stage0] plot failed: {e}", flush=True)

    # Human-readable report.
    lines = []
    lines.append("# Stage 0: MoE dispatch activation low-rank check (Qwen3-30B-A3B)")
    lines.append("")
    lines.append(f"- Model: `{args.model_path}`")
    lines.append(
        f"- hidden_size d = {hidden_size}, num_hidden_layers = {n_layers}, "
        f"num_experts = {cfg.num_experts}, top_k = {cfg.num_experts_per_tok}"
    )
    lines.append(
        f"- Calibration: {len(prompts)} prompts x max_seq_len={args.max_seq_len} "
        f"-> {summary['n_kept_tokens']} non-pad tokens"
    )
    lines.append(f"- Probed layers: {layers_to_hook}")
    lines.append("")
    lines.append("## Cumulative explained variance (centered) at fractions of d")
    lines.append("| layer | d/8 | d/4 | d/2 | rank for 0.90 | rank for 0.95 | rank for 0.99 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for L, info in summary["layers"].items():
        ev = info["explained_variance_centered_at"]
        r = info["rank_at_explained_variance"]
        lines.append(
            f"| {L} | {ev['d/8']:.3f} | {ev['d/4']:.3f} | {ev['d/2']:.3f} | "
            f"{r['0.90']} | {r['0.95']} | {r['0.99']} |"
        )
    lines.append("")
    lines.append(
        "## Go/No-Go (rule from report §6: ≥0.90 explained variance at ℓ = d/4)"
    )
    lines.append("")
    lines.append(f"**Verdict: {summary['go_criteria']['verdict']}**")
    lines.append("")
    for L, v in summary["go_criteria"]["per_layer_explained_variance_at_d_over_4"].items():
        flag = "OK" if v >= 0.90 else "FAIL"
        lines.append(f"- layer {L}: explained variance at d/4 = {v:.3f}  [{flag}]")
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines))
    print(f"[stage0] wrote {out_dir/'report.md'}", flush=True)
    print(f"[stage0] verdict: {summary['go_criteria']['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
