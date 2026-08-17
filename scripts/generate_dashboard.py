"""Regenerates the training results dashboard (loss/accuracy curves, confusion
matrices, summary table) from results/epoch_metrics.json and
results/confusion_summary.json. Run from the repository root:

    python scripts/generate_dashboard.py

Re-run after a new training pass and after regenerating results/epoch_metrics.json /
results/confusion_summary.json (see journal, 2026-08-17 entry, for how those were
first derived from raw training logs and per-run evaluation JSON).
"""

import json
import math
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "results"
epoch_metrics = json.load(open(BASE / "epoch_metrics.json"))
conf_summary = json.load(open(BASE / "confusion_summary.json"))

ARCHS = ["resnet50", "efficientnetb4", "vgg16"]
TASKS = ["seven_class", "binary"]
ARCH_LABEL = {"resnet50": "ResNet-50", "efficientnetb4": "EfficientNetB4", "vgg16": "VGG16"}
TASK_LABEL = {"seven_class": "Seven-class", "binary": "Binary (malignant/benign)"}
SEVEN_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
BINARY_CLASSES = ["benign", "malignant"]

# Categorical slots 1-3 (validated all-pairs, both modes)
ARCH_COLOR_LIGHT = {"resnet50": "#2a78d6", "efficientnetb4": "#eb6834", "vgg16": "#1baf7a"}
ARCH_COLOR_DARK = {"resnet50": "#3987e5", "efficientnetb4": "#d95926", "vgg16": "#199e70"}

# Sequential blue ramp, steps 100..700 (light -> dark)
SEQ_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def seq_color(norm):
    # norm in [0,1] -> nearest ramp step
    idx = min(len(SEQ_RAMP) - 1, max(0, round(norm * (len(SEQ_RAMP) - 1))))
    return SEQ_RAMP[idx]


def text_color_for(norm):
    return "#0b0b0b" if norm < 0.55 else "#ffffff"


# ---- Chart geometry for training curves ----
PANEL_W, PANEL_H = 300, 150
PAD_L, PAD_R, PAD_T, PAD_B = 38, 10, 10, 22
PLOT_W = PANEL_W - PAD_L - PAD_R
PLOT_H = PANEL_H - PAD_T - PAD_B

all_losses = []
all_val_losses = []
for k, epochs in epoch_metrics.items():
    for e in epochs:
        all_losses.append(e["loss"])
        all_val_losses.append(e["val_loss"])
LOSS_MAX = math.ceil(max(all_losses + all_val_losses) * 10) / 10  # round up to nearest 0.1
LOSS_MIN = 0.0
ACC_MIN, ACC_MAX = 0.0, 1.0

N_EPOCHS = 15


def x_for(epoch):
    return PAD_L + (epoch - 1) / (N_EPOCHS - 1) * PLOT_W


def y_for(value, vmin, vmax):
    frac = (value - vmin) / (vmax - vmin)
    return PAD_T + (1 - frac) * PLOT_H


def polyline_points(epochs, key, vmin, vmax):
    pts = [f"{x_for(e['epoch']):.1f},{y_for(e[key], vmin, vmax):.1f}" for e in epochs]
    return " ".join(pts)


def make_panel(arch, task, metric):
    # metric: "accuracy" or "loss"
    key = f"{arch}_{task}"
    epochs = epoch_metrics[key]
    vmin, vmax = (ACC_MIN, ACC_MAX) if metric == "accuracy" else (LOSS_MIN, LOSS_MAX)
    train_key = "accuracy" if metric == "accuracy" else "loss"
    val_key = "val_accuracy" if metric == "accuracy" else "val_loss"
    color_l = ARCH_COLOR_LIGHT[arch]
    color_d = ARCH_COLOR_DARK[arch]

    train_pts = polyline_points(epochs, train_key, vmin, vmax)
    val_pts = polyline_points(epochs, val_key, vmin, vmax)

    # gridlines: 4 horizontal steps
    grid_lines = []
    tick_labels = []
    for i in range(5):
        frac = i / 4
        yv = PAD_T + frac * PLOT_H
        val = vmax - frac * (vmax - vmin)
        grid_lines.append(f'<line x1="{PAD_L}" y1="{yv:.1f}" x2="{PANEL_W - PAD_R}" y2="{yv:.1f}" class="gridline"/>')
        label = f"{val:.1f}" if metric == "loss" else f"{val:.1f}"
        tick_labels.append(f'<text x="{PAD_L - 6}" y="{yv + 3:.1f}" class="tick" text-anchor="end">{label}</text>')

    # phase boundary marker at epoch 5/6 (frozen -> finetune)
    boundary_x = x_for(5.5)
    boundary = f'<line x1="{boundary_x:.1f}" y1="{PAD_T}" x2="{boundary_x:.1f}" y2="{PANEL_H - PAD_B}" class="phase-boundary"/>'

    # invisible hit dots for hover, carrying data via data-* attrs
    dots = []
    for e in epochs:
        cx = x_for(e["epoch"])
        cy_train = y_for(e[train_key], vmin, vmax)
        cy_val = y_for(e[val_key], vmin, vmax)
        dots.append(
            f'<circle cx="{cx:.1f}" cy="{cy_train:.1f}" r="12" class="hit" '
            f'data-epoch="{e["epoch"]}" data-phase="{e["phase"]}" '
            f'data-train="{e[train_key]:.4f}" data-val="{e[val_key]:.4f}" '
            f'data-arch="{ARCH_LABEL[arch]}" data-metric="{metric}"/>'
        )

    svg = f'''
    <svg class="panel-svg metric-{metric}" viewBox="0 0 {PANEL_W} {PANEL_H}" data-arch="{arch}" data-task="{task}">
      {''.join(grid_lines)}
      {boundary}
      <polyline points="{train_pts}" fill="none" stroke="{color_l}" stroke-width="2" class="line-train" style="--c-light:{color_l};--c-dark:{color_d}"/>
      <polyline points="{val_pts}" fill="none" stroke="{color_l}" stroke-width="2" stroke-dasharray="4,3" class="line-val" style="--c-light:{color_l};--c-dark:{color_d}"/>
      {''.join(tick_labels)}
      {''.join(dots)}
    </svg>'''
    return svg


panels_html = []
for task in TASKS:
    for arch in ARCHS:
        acc_svg = make_panel(arch, task, "accuracy")
        loss_svg = make_panel(arch, task, "loss")
        panels_html.append(f'''
        <div class="panel">
          <div class="panel-title">
            <span class="dot" style="background:var(--arch-{arch})"></span>
            {ARCH_LABEL[arch]} <span class="muted">&middot; {TASK_LABEL[task]}</span>
          </div>
          {acc_svg}
          {loss_svg}
        </div>''')

panels_grid = "\n".join(panels_html)

# ---- Confusion matrices ----


def make_confusion(arch, task):
    key = f"{arch}_{task}"
    cm = conf_summary[key]["confusion_matrix"]
    classes = SEVEN_CLASSES if task == "seven_class" else BINARY_CLASSES
    n = len(classes)
    cell = 30 if task == "seven_class" else 46
    label_w = 46
    label_h = 34
    total_w = label_w + n * cell
    total_h = label_h + n * cell
    max_val = max(max(row) for row in cm)

    cells = []
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            norm = v / max_val if max_val else 0
            fill = seq_color(norm)
            tcolor = text_color_for(norm)
            x = label_w + j * cell
            y = label_h + i * cell
            cells.append(
                f'<g class="cm-cell" data-true="{classes[i]}" data-pred="{classes[j]}" data-count="{v}">'
                f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" fill="{fill}" rx="2"/>'
                f'<text x="{x + (cell - 2) / 2:.1f}" y="{y + (cell - 2) / 2 + 4:.1f}" '
                f'text-anchor="middle" fill="{tcolor}" class="cm-count">{v}</text>'
                f'</g>'
            )

    col_labels = []
    for j, c in enumerate(classes):
        x = label_w + j * cell + (cell - 2) / 2
        col_labels.append(
            f'<text x="{x:.1f}" y="{label_h - 6}" text-anchor="middle" class="cm-label">{c}</text>'
        )
    row_labels = []
    for i, c in enumerate(classes):
        y = label_h + i * cell + (cell - 2) / 2 + 4
        row_labels.append(
            f'<text x="{label_w - 6}" y="{y:.1f}" text-anchor="end" class="cm-label">{c}</text>'
        )

    axis_labels = (
        f'<text x="{label_w + n * cell / 2:.1f}" y="12" text-anchor="middle" class="cm-axis">Predicted</text>'
        f'<text x="12" y="{label_h + n * cell / 2:.1f}" text-anchor="middle" class="cm-axis" '
        f'transform="rotate(-90 12 {label_h + n * cell / 2:.1f})">True</text>'
    )

    svg = (
        f'<svg class="cm-svg" viewBox="0 0 {total_w} {total_h + 10}">'
        f'{axis_labels}{"".join(col_labels)}{"".join(row_labels)}{"".join(cells)}'
        f'</svg>'
    )
    acc = conf_summary[key]["accuracy"]
    kappa = conf_summary[key]["kappa"]
    return f'''
    <div class="cm-card">
      <div class="panel-title">
        <span class="dot" style="background:var(--arch-{arch})"></span>
        {ARCH_LABEL[arch]} <span class="muted">&middot; {TASK_LABEL[task]}</span>
      </div>
      {svg}
      <div class="cm-stats">accuracy {acc:.3f} &middot; kappa {kappa:.3f}</div>
    </div>'''


cm_seven = "\n".join(make_confusion(a, "seven_class") for a in ARCHS)
cm_binary = "\n".join(make_confusion(a, "binary") for a in ARCHS)

# ---- Summary table ----
table_rows = []
for arch in ARCHS:
    for task in TASKS:
        key = f"{arch}_{task}"
        d = conf_summary[key]
        isic = f"{d['isic_accuracy']:.3f}" if "isic_accuracy" in d else "&mdash;"
        table_rows.append(f'''
        <tr>
          <td><span class="dot" style="background:var(--arch-{arch})"></span>{ARCH_LABEL[arch]}</td>
          <td>{TASK_LABEL[task]}</td>
          <td class="num">{d['accuracy']:.3f}</td>
          <td class="num">{d['kappa']:.3f}</td>
          <td class="num">{d['roc_auc']:.3f}</td>
          <td class="num">{isic}</td>
          <td class="num">{d['faithfulness']['mean_iou']:.3f}</td>
          <td class="num">{d['faithfulness']['mean_dice']:.3f}</td>
        </tr>''')
table_html = "\n".join(table_rows)

trade_off = json.load(open(BASE / "trade_off_summary.json"))
summary_text = trade_off["summary"].replace("\n", "<br/>")

html = f'''<!doctype html>
<title>Training Results Dashboard</title>
<style>
  .viz-root {{
    color-scheme: light;
    --surface-1: #fcfcfb;
    --page: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --gridline: #e1e0d9;
    --border: rgba(11,11,11,0.10);
    --arch-resnet50: {ARCH_COLOR_LIGHT['resnet50']};
    --arch-efficientnetb4: {ARCH_COLOR_LIGHT['efficientnetb4']};
    --arch-vgg16: {ARCH_COLOR_LIGHT['vgg16']};
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    background: var(--page);
    color: var(--text-primary);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{
      color-scheme: dark;
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --gridline: #2c2c2a;
      --border: rgba(255,255,255,0.10);
      --arch-resnet50: {ARCH_COLOR_DARK['resnet50']};
      --arch-efficientnetb4: {ARCH_COLOR_DARK['efficientnetb4']};
      --arch-vgg16: {ARCH_COLOR_DARK['vgg16']};
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    color-scheme: dark;
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --gridline: #2c2c2a;
    --border: rgba(255,255,255,0.10);
    --arch-resnet50: {ARCH_COLOR_DARK['resnet50']};
    --arch-efficientnetb4: {ARCH_COLOR_DARK['efficientnetb4']};
    --arch-vgg16: {ARCH_COLOR_DARK['vgg16']};
  }}

  * {{ box-sizing: border-box; }}
  body {{ margin: 0; }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 80px; }}
  h1 {{ font-size: 22px; margin: 0 0 4px; }}
  .subtitle {{ color: var(--text-secondary); font-size: 14px; margin: 0 0 28px; }}
  h2 {{ font-size: 16px; margin: 40px 0 4px; }}
  .section-desc {{ color: var(--text-secondary); font-size: 13px; margin: 0 0 16px; max-width: 720px; }}
  .muted {{ color: var(--text-muted); font-weight: 400; }}

  .legend {{ display: flex; gap: 18px; margin-bottom: 16px; font-size: 13px; color: var(--text-secondary); flex-wrap: wrap; align-items: center; }}
  .legend .dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; margin-right: 6px; }}
  .legend .key {{ display: inline-flex; align-items: center; }}
  .legend .lkey {{ display: inline-block; width: 18px; height: 0; border-top: 2px solid var(--text-secondary); margin-right: 6px; vertical-align: middle; }}
  .legend .lkey.dashed {{ border-top-style: dashed; }}

  .toggle {{ display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; margin-bottom: 18px; }}
  .toggle input {{ display: none; }}
  .toggle label {{ padding: 6px 16px; font-size: 13px; cursor: pointer; color: var(--text-secondary); user-select: none; }}
  .toggle input:checked + label {{ background: var(--text-primary); color: var(--surface-1); }}

  .panels-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
  .panel {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px 8px; }}
  .panel-title {{ font-size: 13px; font-weight: 600; margin-bottom: 6px; display: flex; align-items: center; gap: 6px; }}
  .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}
  .panel-svg {{ width: 100%; height: auto; display: none; }}
  #metric-accuracy:checked ~ .panels-grid .metric-accuracy {{ display: block; }}
  #metric-loss:checked ~ .panels-grid .metric-loss {{ display: block; }}

  .gridline {{ stroke: var(--gridline); stroke-width: 1; }}
  .phase-boundary {{ stroke: var(--text-muted); stroke-width: 1; stroke-dasharray: 2,2; opacity: 0.6; }}
  .tick {{ font-size: 8px; fill: var(--text-muted); font-variant-numeric: tabular-nums; }}
  .line-train {{ stroke: var(--c-light); }}
  .line-val {{ stroke: var(--c-light); }}
  :root[data-theme="dark"] .line-train, :root[data-theme="dark"] .line-val {{ stroke: var(--c-dark); }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .line-train,
    :root:where(:not([data-theme="light"])) .line-val {{ stroke: var(--c-dark); }}
  }}
  .hit {{ fill: transparent; cursor: crosshair; }}

  .cm-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }}
  .cm-card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }}
  .cm-svg {{ width: 100%; height: auto; }}
  .cm-label {{ font-size: 8px; fill: var(--text-secondary); }}
  .cm-axis {{ font-size: 9px; fill: var(--text-muted); }}
  .cm-count {{ font-size: 9px; font-variant-numeric: tabular-nums; pointer-events: none; }}
  .cm-stats {{ font-size: 12px; color: var(--text-secondary); margin-top: 8px; font-variant-numeric: tabular-nums; }}
  .cm-cell {{ cursor: pointer; }}
  .cm-cell rect {{ transition: opacity .1s; }}
  .cm-cell:hover rect {{ opacity: 0.75; stroke: var(--text-primary); stroke-width: 1.5; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }}
  th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--gridline); }}
  th {{ color: var(--text-muted); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: .03em; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:last-child td {{ border-bottom: none; }}
  td .dot {{ margin-right: 8px; }}

  .callout {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 16px 18px; font-size: 13px; line-height: 1.6; color: var(--text-secondary); margin-top: 16px; }}
  .callout strong {{ color: var(--text-primary); }}

  #tooltip {{ position: fixed; pointer-events: none; background: var(--text-primary); color: var(--surface-1); font-size: 12px; padding: 6px 10px; border-radius: 6px; opacity: 0; transition: opacity .08s; z-index: 10; white-space: nowrap; }}
  #tooltip .tt-val {{ font-weight: 600; font-variant-numeric: tabular-nums; }}

  @media (max-width: 900px) {{
    .panels-grid, .cm-grid {{ grid-template-columns: 1fr 1fr; }}
  }}
  @media (max-width: 600px) {{
    .panels-grid, .cm-grid {{ grid-template-columns: 1fr; }}
  }}
</style>

<div class="viz-root">
  <div class="wrap">
    <h1>Skin Cancer CNN &mdash; Training Results</h1>
    <p class="subtitle">3 architectures &times; 2 tasks &middot; 15 epochs each (5 frozen warm-up + 10 fine-tune) &middot; Runpod RTX 4090 &middot; run completed 2026-08-17</p>

    <h2>Training curves</h2>
    <p class="section-desc">Solid line = training, dashed = validation. Dotted vertical marker = transition from frozen-backbone warm-up to fine-tuning. Small multiples share one y-axis per metric so architectures are directly comparable.</p>
    <div class="legend">
      <span class="key"><span class="lkey"></span>Training</span>
      <span class="key"><span class="lkey dashed"></span>Validation</span>
      <span class="key"><span class="dot" style="background:var(--arch-resnet50)"></span>ResNet-50</span>
      <span class="key"><span class="dot" style="background:var(--arch-efficientnetb4)"></span>EfficientNetB4</span>
      <span class="key"><span class="dot" style="background:var(--arch-vgg16)"></span>VGG16</span>
    </div>
    <div class="toggle">
      <input type="radio" name="metric" id="metric-accuracy" checked>
      <label for="metric-accuracy">Accuracy</label>
      <input type="radio" name="metric" id="metric-loss">
      <label for="metric-loss">Loss</label>
    </div>
    <div class="panels-grid">
      {panels_grid}
    </div>

    <h2>Confusion matrices &mdash; seven-class task</h2>
    <p class="section-desc">Internal validation split. Rows = true class, columns = predicted class. Darker = higher count within that matrix.</p>
    <div class="cm-grid">
      {cm_seven}
    </div>

    <h2>Confusion matrices &mdash; binary task</h2>
    <p class="section-desc">Internal validation split (HAM10000 + DDI combined).</p>
    <div class="cm-grid">
      {cm_binary}
    </div>

    <h2>Summary metrics</h2>
    <table>
      <thead>
        <tr>
          <th>Architecture</th><th>Task</th><th class="num">Accuracy</th><th class="num">Kappa</th>
          <th class="num">ROC-AUC</th><th class="num">ISIC2018 acc.</th><th class="num">Grad-CAM IoU</th><th class="num">Grad-CAM Dice</th>
        </tr>
      </thead>
      <tbody>
        {table_html}
      </tbody>
    </table>
    <div class="callout">{summary_text}</div>
  </div>
</div>

<div id="tooltip"></div>
<script>
  const tooltip = document.getElementById('tooltip');

  document.querySelectorAll('.hit').forEach(dot => {{
    dot.addEventListener('pointerenter', (ev) => {{
      const d = dot.dataset;
      const metricName = d.metric === 'accuracy' ? 'accuracy' : 'loss';
      const fmt = v => d.metric === 'accuracy' ? (parseFloat(v)*100).toFixed(1) + '%' : parseFloat(v).toFixed(4);
      tooltip.innerHTML = `<div>${{d.arch}} &middot; epoch ${{d.epoch}} (${{d.phase}})</div>` +
        `<div>train ${{metricName}}: <span class="tt-val">${{fmt(d.train)}}</span></div>` +
        `<div>val ${{metricName}}: <span class="tt-val">${{fmt(d.val)}}</span></div>`;
      tooltip.style.opacity = 1;
    }});
    dot.addEventListener('pointermove', (ev) => {{
      tooltip.style.left = (ev.clientX + 14) + 'px';
      tooltip.style.top = (ev.clientY + 14) + 'px';
    }});
    dot.addEventListener('pointerleave', () => {{ tooltip.style.opacity = 0; }});
  }});

  document.querySelectorAll('.cm-cell').forEach(cell => {{
    cell.addEventListener('pointerenter', () => {{
      const d = cell.dataset;
      tooltip.innerHTML = `<div>true: <span class="tt-val">${{d.true}}</span></div>` +
        `<div>predicted: <span class="tt-val">${{d.pred}}</span></div>` +
        `<div>count: <span class="tt-val">${{d.count}}</span></div>`;
      tooltip.style.opacity = 1;
    }});
    cell.addEventListener('pointermove', (ev) => {{
      tooltip.style.left = (ev.clientX + 14) + 'px';
      tooltip.style.top = (ev.clientY + 14) + 'px';
    }});
    cell.addEventListener('pointerleave', () => {{ tooltip.style.opacity = 0; }});
  }});
</script>
'''

out_path = BASE / "training_dashboard.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(html)
print("wrote", out_path, len(html), "bytes")
print("LOSS_MAX", LOSS_MAX)
