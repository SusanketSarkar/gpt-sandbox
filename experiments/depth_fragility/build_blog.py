"""Build the depth-fragility blog (index.html) from results.json + blog_template.html.

Data-driven parts (tables, stats, verdict booleans) are computed here from results.json.
Narrative text is taken from narrative.json if present (written by hand after seeing the
results), otherwise auto-derived from the numbers.
"""
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "results.json").read_text())
SIZES = R["sizes"]
NARR = json.loads((HERE / "narrative.json").read_text()) if (HERE / "narrative.json").exists() else {}


def fmt(x, nd=4):
    return "—" if x is None else f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def esc(s):
    return html.escape(str(s))


def stat(v, l, cls=""):
    return f'<div class="stat"><div class="v {cls}">{v}</div><div class="l">{l}</div></div>'


def curve_table(size, s):
    rows = []
    for key in sorted(s["fragility"]["curve"], key=lambda k: (s["fragility"]["curve"][k]["sigma"], str(s["fragility"]["curve"][k]["layer"]))):
        c = s["fragility"]["curve"][key]
        loc = f"layer {c['layer']}" if isinstance(c["layer"], int) else "ALL layers"
        rows.append(f"<tr><td>{key}</td><td>{loc}</td><td>{c['sigma']}</td>"
                    f"<td>{c['bpb_mean']:.4f}</td><td>{c['bpb_std']:.4f}</td>"
                    f"<td><b>{c['delta']:+.4f}</b></td><td>{c['delta_over_floor']:+.2f}×</td>"
                    f"<td>{c.get('gen_gap_mean') if c.get('gen_gap_mean') is not None else '—'}</td>"
                    f"<td>{c['n_seeds']}</td></tr>")
    return f"""<table><thead><tr><th>variant</th><th>location</th><th>sigma</th><th>bpb</th><th>bpb σ</th><th>Δ vs plain</th><th>Δ/floor</th><th>gen_gap</th><th>seeds</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>"""


def anchor_block(size, s):
    at = s["anchor_test"]
    if not at:
        return ""
    rows = []
    for k in sorted(at):
        r = at[k]
        sf, sl = r["signs_mid_worse_than_first"], r["signs_mid_worse_than_last"]
        rows.append(
            f"<tr><td>{k}</td><td>{fmt(r['delta_first'])}</td><td>{fmt(r['delta_mid'])}</td><td>{fmt(r['delta_last'])}</td>"
            f"<td><b>{fmt(r['d_first_minus_mid_mean'])}</b></td><td>{fmt(r['d_last_minus_mid_mean'])}</td>"
            f"<td>{'mid worse' if sum(sf) > len(sf)/2 else ('first worse' if sum(sf) < len(sf)/2 else 'mixed')}</td>"
            f"<td>{'mid worse' if sum(sl) < len(sl)/2 else ('last worse' if sum(sl) > len(sl)/2 else 'mixed')}</td>"
            f"<td>{fmt(r['in_floor_units_first_minus_mid'], 2)}×</td><td>{fmt(r['in_floor_units_last_minus_mid'], 2)}×</td></tr>")
    return f"""<table><thead><tr><th>sigma</th><th>Δ first</th><th>Δ mid</th><th>Δ last</th><th>d(first)−d(mid)</th><th>d(last)−d(mid)</th><th>mid vs first</th><th>mid vs last</th><th>|Δ|/floor (f−m)</th><th>|Δ|/floor (l−m)</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="small muted">Paired over the 3 core seeds where available. d(first)−d(mid) &gt; 0 = middle more fragile (hypothesis);
d(last)−d(mid) &gt; 0 = last more fragile than middle (against it).</p>"""


def shape_block(size, s):
    st = s["shape_test"]
    if not st:
        return ""
    rows = []
    for k in sorted(st):
        r = st[k]
        def sig(v):
            return f"{v:+.3f} (p={r['p_perm_' + 'linear' if 'linear' in v else ('gauss' if 'gauss' in v else 'depth')]:.3f})"
        rows.append(f"<tr><td>{k}</td><td><b>{r['spearman_linear_bump']:+.3f}</b></td><td>{r['p_perm_linear']:.3f}</td>"
                    f"<td>{r['spearman_gauss_bump']:+.3f}</td><td>{r['p_perm_gauss']:.3f}</td>"
                    f"<td>{r['spearman_depth']:+.3f}</td><td>{r['p_perm_depth']:.3f}</td></tr>")
    return f"""<table><thead><tr><th>curve</th><th>ρ vs linear bump</th><th>p</th><th>ρ vs gauss bump</th><th>p</th><th>ρ vs depth</th><th>p</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="small muted">Bump centrality: 1 at the middle layer, 0 at the ends (linear and Gaussian variants).
p = permutation test (2000 shuffles of the curve against the prior). A hypothesis-faithful result: strong positive ρ
vs the bump, weak vs raw depth.</p>"""


def abla_block(size, s):
    ac = s["ablation_corr"]
    if not ac:
        return "<p class='muted small'>No ablation profiles available yet for this size.</p>"
    rows = []
    for k in sorted(ac):
        r = ac[k]
        rows.append(f"<tr><td>{k}</td><td><b>{fmt(r['rho_fragility_ablation'], 3)}</b></td><td>{fmt(r['p_perm_fragility_ablation'], 3)}</td>"
                    f"<td>{fmt(r.get('rho_fragility_lens'), 3)}</td><td>{fmt(r.get('rho_fragility_scale'), 3)}</td>"
                    f"<td>{fmt(r.get('rho_ablation_scale'), 3)}</td></tr>")
    return f"""<table><thead><tr><th>curve</th><th>ρ(fragility, ablate dloss)</th><th>p</th><th>ρ(fragility, lens marginal)</th><th>ρ(fragility, out-proj scale)</th><th>ρ(ablate, scale)</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="small muted">If fragility tracks <i>importance</i> (ablation/lens), that is a density story. If it tracks
<i>output scale</i> more strongly, that is a bookkeeping story (bigger writers feel their noise more). The
ablate↔scale column tells you how much of the importance curve is just scale.</p>"""


def sens_block(size, s):
    sens = s.get("sensitivity")
    if not sens:
        return f"<p class='muted small'>{s['label']}: test-time sensitivity profile not computed yet.</p>"
    import os
    has_plot = os.path.exists(HERE / "plots" / f"sensitivity_{size}.png")
    fig = ""
    if has_plot:
        fig = f"""<figure>
      <img src="plots/sensitivity_{size}.png" alt="sensitivity {size}">
      <figcaption>Red: train-time fragility (trained with noise at L, evaluated clean). Blue: test-time
      sensitivity of the plain model (evaluated with noise at L). Both Δval_bpb at σ=1.0.</figcaption>
    </figure>"""
    rows = []
    for k, r in sorted(sens["per_sigma"].items()):
        def g(rr, kk):
            v = rr.get(kk)
            return "—" if v is None else f"{v:+.3f}"
        rows.append(f"<tr><td>{k}</td><td>{g(r, 'spearman_linear_bump')}</td><td>{r['p_perm_linear']:.3f}</td>"
                    f"<td>{g(r, 'spearman_gauss_bump')}</td><td>{r['p_perm_gauss']:.3f}</td>"
                    f"<td><b>{r['peak_layer']}</b></td><td>{r['min_layer']}</td>"
                    f"<td>{g(r, 'rho_vs_ablation')}</td><td>{'—' if r.get('p_vs_ablation') is None else format(r['p_vs_ablation'], '.3f')}</td>"
                    f"<td>{g(r, 'rho_vs_lens_marginal')}</td><td>{g(r, 'rho_vs_scale')}</td>"
                    f"<td><b>{g(r, 'rho_vs_train_fragility')}</b></td></tr>")
    table = f"""<table><thead><tr><th>σ</th><th>ρ vs linear bump</th><th>p</th><th>ρ vs gauss bump</th><th>p</th><th>peak L</th><th>min L</th><th>ρ vs ablation</th><th>p</th><th>ρ vs lens</th><th>ρ vs out-proj scale</th><th>ρ vs train-time fragility</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="small muted">Sensitivity = Δval_bpb of the plain model when one stream slice is noised at eval time.
Peak/min = the most/least sensitive layer. Correlations test which story fits: a density story predicts high
ρ with ablation/lens; a bookkeeping story predicts high ρ with output-projection scale.</p>"""
    return fig + table


def reg_block(size, s):
    reg = s["regulariser"]
    parts = []
    beats = reg["variants_beating_plain"] or "none"
    parts.append(f"<p><b>Plain gen_gap:</b> {reg['plain_gen_gap']:+.4f} nats. "
                 f"<b>Variants beating plain by &gt;1 floor:</b> {esc(beats)}.</p>")
    gg = reg["gen_gap_by_variant"]
    if gg:
        rows = "".join(f"<tr><td>{k}</td><td>{v if v is not None else '—'}</td></tr>" for k, v in sorted(gg.items()))
        parts.append(f"<table><thead><tr><th>variant</th><th>gen_gap (nats)</th></tr></thead><tbody>{rows}</tbody></table>")
    for k, v in reg.items():
        if k.startswith("global_vs_mid"):
            ratio = f"{v['ratio_global_mid']:.2f}×" if v["ratio_global_mid"] is not None else "n/a"
            parts.append(f"<p><b>{k}:</b> global Δ = {v['delta_global']:+.4f}, mid-only Δ = {v['delta_mid']:+.4f} → "
                         f"global is {ratio} the mid effect (pure-dose expectation ≈ {v['expected_ratio_if_pure_dose']:.0f}× if damage adds per layer). "
                         f"A ratio far below pure dose means per-layer damage does not add — location matters.</p>")
    dyn = s["dynamics"]
    if dyn.get("plain_final_train") is not None:
        parts.append(f"<p>Plain ends at train loss {dyn['plain_final_train']:.4f} / val {dyn['plain_final_val']:.4f}. "
                     f"Noise variants ending at higher train loss are consistent with noise capping what the model memorises "
                     f"(a regulariser); ending at similar train loss but worse val is a capability loss (density).</p>")
    return "".join(parts)


def floor_stats():
    out = []
    for size, s in SIZES.items():
        nf = s["noise_floor"]
        out.append(stat(f"{nf['val_bpb_mean']:.4f}", f"plain val_bpb · {s['label']} (n={nf['n']} seeds)"))
        out.append(stat(f"±{nf['val_bpb_std']:.4f}", f"seed noise floor · {s['label']}"))
    return "".join(out)


def sweep_table():
    rows = [
        ("core", "plain + first/mid/last × σ∈{0.1,0.3,1.0}", "3", "30 per size", "the hypothesis test"),
        ("profile", "every layer L × σ∈{0.3,1.0}", "1", "16 per size", "the full fragility curve"),
        ("global", "all layers × σ∈{0.1,0.3,1.0}", "1", "3 per size", "pure-dose / regulariser control"),
    ]
    body = "".join(f"<tr><td>{a} <span class='muted small'>({b})</span></td><td>{c}</td><td>{d}</td><td>{e}</td></tr>" for a, b, c, d, e in rows)
    return f"""<table><thead><tr><th>cell</th><th>variants</th><th>seeds</th><th>runs</th><th>purpose</th></tr></thead><tbody>{body}</tbody></table>"""


def verdict(size, s):
    """Auto-derive a verdict for one size from a 2-D matrix:

    train-time fragility profile (x) test-time sensitivity profile:
      bump x bump   -> middle-more-fragile (density story holds in both senses)
      bump x flat   -> adaptation-only (training effect, not static importance)
      flat x bump   -> sensitivity-only (model depends on mid, noise is a generic regulariser)
      flat x flat   -> no-clear-depth-structure (noise acts generically)
    """
    st = s["shape_test"].get("s100") or s["shape_test"].get("s30") or {}
    rho, p = st.get("spearman_linear_bump"), st.get("p_perm_linear")
    at = s.get("anchor_test") or {}
    key = sorted(at)[-1] if at else None
    a = at.get(key, {}) if key else {}
    mid_worse = bool(a and ((a.get("d_first_minus_mid_mean") or 0) > 0 or (a.get("d_last_minus_mid_mean") or 0) < 0))
    train = "bump" if (rho is not None and p is not None and rho > 0.5 and p < 0.1 and mid_worse) else ("flat" if rho is not None else "n/a")
    ss = (s.get("sensitivity") or {}).get("per_sigma", {})
    s100 = ss.get("sig100") or ss.get("sig30") or {}
    test = "bump" if (s100.get("spearman_linear_bump") is not None
                      and s100["spearman_linear_bump"] > 0.5 and s100.get("p_perm_linear", 1) < 0.2) \
        else ("flat" if s100 else "n/a")
    if train == "bump" and test in ("bump", "n/a"):
        return "middle-more-fragile (density story)"
    if train == "bump" and test == "flat":
        return "adaptation-only (train-time effect)"
    if train == "flat" and test == "bump":
        return "sensitivity-only (generic regulariser)"
    if train == "flat" and test == "flat":
        return "no-clear-depth-structure"
    return "incomplete (more data needed)"


def main():
    total_runs = sum(s["n_runs"] for s in SIZES.values())
    t = (HERE / "blog_template.html").read_text()

    # pills
    pills = [
        f'<span class="pill"><b>{total_runs}</b> runs</span>',
        f'<span class="pill"><b>2</b> sizes (8L / 16L)</span>',
        f'<span class="pill"><b>351.4M</b> tokens</span>',
        f'<span class="pill"><b>1.56M</b> corpus tokens</span>',
        f'<span class="pill"><b>3</b>× H100</span>',
        f'<span class="pill">~6h wall</span>',
    ]
    for size, s in SIZES.items():
        pills.append(f'<span class="pill"><b>{s["n_layer"]}</b> layers</span>')
    pills = "".join(dict.fromkeys(pills))

    sub = {
        "{{PILLS}}": pills,
        "{{FLOOR_STATS}}": floor_stats(),
        "{{SWEEP_TABLE}}": sweep_table(),
        "{{TOTAL_RUNS}}": str(total_runs),
        "{{SCALE_NOTE}}": NARR.get("scale_note", "filled in after results land."),
    }
    sub["{{SENS_BLOCKS}}"] = "".join(f"<h4>{s['label']}</h4>" + sens_block(size, s) for size, s in SIZES.items())
    sub["{{ANCHOR_BLOCKS}}"] = "".join(f"<h3>{s['label']}</h3>" + anchor_block(size, s) for size, s in SIZES.items())
    sub["{{SHAPE_BLOCKS}}"] = "".join(shape_block(size, s) for size, s in SIZES.items())
    sub["{{ABLA_BLOCKS}}"] = "".join(abla_block(size, s) for size, s in SIZES.items())
    sub["{{REG_BLOCKS}}"] = "".join(f"<h3>{s['label']}</h3>" + reg_block(size, s) for size, s in SIZES.items())

    # narrative (hand-written in narrative.json after results land), with auto-fallback
    vm = verdict("depth_m", SIZES.get("depth_m", {})) if "depth_m" in SIZES else None
    vd = verdict("depth_d", SIZES.get("depth_d", {})) if "depth_d" in SIZES else None
    auto = {
        "deck": NARR.get("deck", "A 114-run study of where Gaussian noise in the residual stream does the most damage, "
                                 "injected at every depth of two GPTs trained on the same 1.56M-token corpus."),
        "verdict_tag": NARR.get("verdict_tag", "auto-deriving from results.json"),
        "verdict_tagclass": NARR.get("verdict_tagclass", "blue"),
        "verdict_text": NARR.get("verdict_text", f"M: {vm} · D: {vd} (auto verdict — run build_blog again after narrative.json is written)."),
        "answer_tag": NARR.get("answer_tag", "auto"),
        "answer_tagclass": NARR.get("answer_tagclass", "blue"),
        "answer_class": NARR.get("answer_class", "hypo"),
        "answer_text": NARR.get("answer_text", "Auto: see results.json / results.md for the full numbers."),
        "caveats": NARR.get("caveats", "<p><i>Written after the results land.</i></p>"),
    }
    sub["{{DECK}}"] = auto["deck"]
    sub["{{VERDICT_TAG}}"] = auto["verdict_tag"]
    sub["{{VERDICT_TAGCLASS}}"] = auto["verdict_tagclass"]
    sub["{{VERDICT_TEXT}}"] = auto["verdict_text"]
    sub["{{ANSWER_TAG}}"] = auto["answer_tag"]
    sub["{{ANSWER_TAGCLASS}}"] = auto["answer_tagclass"]
    sub["{{ANSWER_CLASS}}"] = auto["answer_class"]
    sub["{{ANSWER_TEXT}}"] = auto["answer_text"]
    sub["{{CAVEATS}}"] = auto["caveats"]

    out = t
    for k, v in sub.items():
        out = out.replace(k, v)
    (HERE / "index.html").write_text(out)
    leftover = [tok for tok in ("{{DECK}}", "{{PILLS}}", "{{VERDICT_TEXT}}") if tok in out]
    print(f"wrote {HERE/'index.html'} ({len(out)} chars) leftover placeholders: {leftover or 'none'}")


if __name__ == "__main__":
    main()
