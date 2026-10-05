"""Build the bilingual static research site from local public facts (standard library)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "showcase"
sys.path.insert(0, str(ROOT / "src"))
from product_matching.data import business_record, native_record, compact_native
from product_matching.features import pair_features, local_code_features, native_features
from product_matching.normalization import normalize_text, parse_price
PAGES = ("index", "method", "results", "reproduce")
LANGUAGES = ("en", "zh-CN")
METHODS = ("B22", "L30", "S33", "L42", "C+", "C0")
DOCS = ("PROJECT_CARD", "DATA_CARD", "MODEL_CARD", "CORE_USAGE", "REPRODUCIBILITY", "LIMITATIONS", "CODE_MAP")
GITHUB = "https://github.com/Yemyu/product-entity-matching"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def facts_context(results: dict, model: dict, features: dict) -> dict:
    metric = results["metrics"]["C+"]
    context = {
        "evaluation": str(results["roles"]["evaluation"]),
        "positive": str(metric["positive_count"]),
        "features": str(len(features["names"])),
        "ap4": f'{metric["ap"]:.4f}',
        "f14": f'{metric["f1"]:.4f}',
        "p100": f'{metric["p_at_100"]:.2f}',
        "delta_ap": f'{100 * (metric["ap"] - results["metrics"]["L42"]["ap"]):.2f}',
        "delta_f1": f'{100 * (metric["f1"] - results["metrics"]["L42"]["f1"]):.2f}',
        "delta_c0_ap": f'{100 * (metric["ap"] - results["metrics"]["C0"]["ap"]):.4f}',
        "threshold": str(metric["threshold"]),
        "precision4": f'{metric["precision"]:.4f}',
        "recall4": f'{metric["recall"]:.4f}',
        "epoch": str(model["selected_epoch"]),
        "seeds": ", ".join(map(str, model["seeds"])),
        "dropout": str(model["dropout"]),
        "replay_commands": str(results["reproduction"]["fixed_artifact_replay_cli_commands"]),
        "replay_groups": str(results["reproduction"]["independently_checked_result_groups"]),
        "max_difference": str(results["reproduction"]["max_prediction_difference"]),
        "negative": str(metric["count"] - metric["positive_count"]),
        "reference_fp": str(results["metrics"]["L42"]["confusion"]["fp"]),
        "reference_fn": str(results["metrics"]["L42"]["confusion"]["fn"]),
    }
    context.update({key: str(value) for key, value in metric["confusion"].items()})
    return context


class Renderer:
    def __init__(self, language: str, content: dict, results: dict, model: dict, features: dict):
        self.language = language
        self.results, self.model, self.features = results, model, features
        self.context = facts_context(results, model, features)
        def resolve(value):
            if isinstance(value, str):
                return self.fill(value)
            if isinstance(value, list):
                return [resolve(item) for item in value]
            if isinstance(value, dict):
                return {key: resolve(item) for key, item in value.items()}
            return value
        self.content = resolve(content)
        self.other = "zh-CN" if language == "en" else "en"

    def fill(self, value: str) -> str:
        return re.sub(r"\[\[([a-zA-Z0-9_]+)\]\]", lambda match: self.context[match[1]], value)

    def t(self, key: str) -> str:
        return esc(self.fill(self.content[key]))

    def tr(self, chinese: str, english: str) -> str:
        return esc(chinese if self.language == "zh-CN" else english)

    def hero_facts(self) -> str:
        facts = [("AP", self.context['ap4']), ("F1", self.context['f14']),
                 (self.tr("固定评价商品对","Fixed evaluation pairs"),self.context['evaluation'])]
        items = ''.join(f'<div><span>{label}</span><strong>{value}</strong></div>' for label,value in facts)
        return f'<div class="hero-facts">{items}<p>{self.tr("Walmart–Amazon · 回顾性固定评价池","Walmart–Amazon · fixed retrospective evaluation pool")}</p></div>'

    def workflow(self) -> str:
        steps = ''.join(f'<li><span>{i+1:02}</span><strong>{esc(title)}</strong></li>'
                        for i, title in enumerate(self.content["workflow_titles"]))
        return f'<ol class="workflow-strip">{steps}</ol>'

    def normalization_examples(self) -> str:
        rows = [("text", "ＡＣＭＥ  ZX-100", normalize_text("ＡＣＭＥ  ZX-100"), "present")]
        for raw in ("1.999e1", "$19.99", "19,99", None):
            result = parse_price(raw)
            rows.append(("price",raw,result.normalized,result.status))
        body = ''.join(f'<tr><th scope="row"><code>{field}</code></th><td><code>{esc(raw) if raw is not None else "null"}</code></td><td><code>{esc(normalized) if normalized is not None else "null"}</code></td><td><code>{state}</code></td></tr>' for field,raw,normalized,state in rows)
        heads = ''.join(f'<th scope="col">{self.tr(z,e)}</th>' for z,e in [('字段','Field'),('原值','Raw value'),('比较视图','Comparison view'),('质量状态','Quality state')])
        note = self.t("normalization_example_note")
        return f'<div class="table-wrap"><table><thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table></div><p class="caption">{note}</p>'

    def workbench(self) -> str:
        """Run only pure feature calculations on invented records; no model/IDF fitting."""
        cases = [
            ("spelling", "ZX100", "ACME", "89.00", "USD"),
            ("digits", "ZX101", "ACME", "89.00", "USD"),
            ("missing", "ZX100", None, "89.00", "EUR"),
        ]
        labels = self.content["example_feature_labels"]
        tabs, panels = [], []
        for identity, model, brand, price, currency in cases:
            left = {"title":"Acme ZX-100 cordless drill","brand":"Acme","price":"89.00","priceCurrency":"USD"}
            right = {"title":f"ACME {model} cordless drill","brand":brand,"price":price,"priceCurrency":currency}
            a, b = business_record(left), business_record(right)
            an = native_record({"modelno":"ZX-100","category":"power tools"})
            bn = native_record({"modelno":model,"category":"power tools"})
            values = {**pair_features(a,b), **local_code_features(a['normalized']['title'],b['normalized']['title']),
                      **native_features(an,bn,a['normalized']['title'],b['normalized']['title'])}
            title = esc(self.content["example_cases"][identity]["title"])
            tabs.append(f'<button type="button" data-example-case="{identity}" aria-pressed="false">{title}</button>')
            field_rows = []
            for key, z, e in [("title","标题","Title"),("brand","品牌","Brand"),("price","价格","Price"),("priceCurrency","币种","Currency")]:
                display = []
                for raw, view in [(left,a),(right,b)]:
                    raw_value = self.tr("缺失","Missing") if raw.get(key) is None else esc(raw[key])
                    normalized = self.tr("缺失","Missing") if view['normalized'][key] is None else esc(view['normalized'][key])
                    display.append(f'<td><span class="raw-value">{raw_value}</span><span class="normalized-value">→ <code>{normalized}</code></span></td>')
                field_rows.append(f'<tr><th scope="row">{self.tr(z,e)}</th>{"".join(display)}</tr>')
            field_rows.append(f'<tr><th scope="row">{self.tr("原生型号 → 紧凑形式","Native model → compact form")}</th><td>ZX-100 → <code>{compact_native(an["modelno"])}</code></td><td>{model} → <code>{compact_native(bn["modelno"])}</code></td></tr>')
            fields = f'<p class="table-scroll-hint">{self.tr("窄屏可横向滑动查看两条记录；键盘可聚焦表格后使用方向键。","On narrow screens, scroll horizontally to compare both records; keyboard users can focus the table and use arrow keys.")}</p><div class="table-wrap field-comparison record-comparison" tabindex="0" role="region" aria-label="{self.tr("商品字段对照","Product-field comparison")}"><table><thead><tr><th>{self.tr("字段","Field")}</th><th>{self.tr("记录A：原值 → 规范化","Record A: raw → normalized")}</th><th>{self.tr("记录B：原值 → 规范化","Record B: raw → normalized")}</th></tr></thead><tbody>{"".join(field_rows)}</tbody></table></div>'
            rows = ''.join(f'<tr data-demo-feature="{name}" data-demo-value="{values[name]}"><th scope="row">{esc(label)}<code>{name}</code></th><td>{values[name]:.4f}</td></tr>' for name,label in labels.items())
            features = f'<div class="table-wrap field-comparison"><table><thead><tr><th>{self.tr("选取的比较特征","Selected comparison features")}</th><th>{self.tr("程序计算值","Computed value")}</th></tr></thead><tbody>{rows}</tbody></table></div>'
            note = esc(self.content["example_cases"][identity]["note"])
            panels.append(f'<article id="example-{identity}" data-example-panel="{identity}" aria-labelledby="example-title-{identity}"><h3 id="example-title-{identity}">{title}</h3>{fields}{features}<p class="example-explanation">{note}</p></article>')
        note = self.t("example_boundary")
        return f'<div class="field-workbench"><p class="lead-note">{note}</p><div class="example-controls" role="group" aria-label="{self.tr("比较场景","Comparison scenarios")}">{"".join(tabs)}</div>{"".join(panels)}</div>'

    def p(self, key: str, css: str = "") -> str:
        attribute = f' class="{css}"' if css else ""
        return f'<p{attribute}>{self.t(key)}</p>'

    def doc(self, name: str) -> str:
        directory = GITHUB + "/blob/main/docs/" + ("zh-CN/" if self.language == "zh-CN" else "")
        return directory + name + ".md"

    def section(self, anchor: str, title: str, body: str, number: str, tint: bool = False) -> str:
        css = "section section-rule" + (" section-tint" if tint else "")
        return f'<section id="{anchor}" class="{css}"><div class="container"><div class="section-heading"><span class="section-number" aria-hidden="true">{number}</span><h2 class="title is-3">{self.t(title)}</h2></div>{body}</div></section>'

    def link(self, url: str, text: str, css: str = "text-link", download: bool = False) -> str:
        if url.startswith(GITHUB) and "GitHub" not in text:
            text += "（GitHub）" if self.language == "zh-CN" else " (GitHub)"
        label = esc(text)
        if css == "resource-link":
            label = f'<span>{label}</span><span aria-hidden="true">{"↓" if download else "↗"}</span>'
        return f'<a href="{esc(url)}" class="{css}"{ " download" if download else ""}>{label}</a>'

    def details(self, key: str, body: str, identity: str = "") -> str:
        attribute = f' id="{identity}"' if identity else ""
        return f'<details class="reader-details"{attribute}><summary>{self.t(key)}</summary><div class="details-body">{body}</div></details>'

    def overview(self) -> str:
        brief = '<div class="project-brief"><div class="brief-grid">' + ''.join(f'<article class="brief-block"><h3>{esc(item["title"])}</h3><p>{esc(item["text"])}</p></article>' for item in self.content["overview_brief"]) + '</div></div>'
        pipeline = self.p("model_decision_explanation", "section-intro") + self.workflow() + self.link("method.html", self.content["read_method"])
        example = self.p("data_example_intro", "section-intro") + self.workbench()
        metrics = self.results["metrics"]["C+"]
        summaries = []
        for key, label, context in (("ap", "AP", "ap"), ("f1", "F1", "f1"), ("p_at_100", "P@100", "p100")):
            value = metrics[key]
            formatted = f"{value:.2f}" if key == "p_at_100" else f"{value:.4f}"
            summaries.append(f'<div class="summary-metric"><p class="summary-label"><abbr title="{self.t(context+"_expansion")}">{label}</abbr></p><p class="summary-value" data-source="metrics.C+.{key}" data-value="{value}">{formatted}</p><p class="summary-context">{self.t(context+"_context")}</p></div>')
        result = f'<p class="note-kicker">{self.t("evaluation_context")}</p><div class="results-summary">{"".join(summaries)}</div>{self.p("result_interpretation", "section-intro")}{self.p("scope", "lead-note")}{self.link("results.html",self.content["read_results"])}'
        close = f'<div class="prose">{self.p("overview_close_p")}{self.link("reproduce.html#quickstart",self.content["read_reproduce"])}</div>'
        return self.section("project", "brief_title", brief, "01") + self.section("example", "overview_example_title", example, "02") + self.section("evidence", "overview_choice_title", pipeline, "03") + self.section("summary", "overview_result_title", result, "04") + self.section("public-code", "overview_close_title", close, "05")

    def diagram(self) -> str:
        def node(item):
            return f'<div class="architecture-node"><strong>{esc(item["title"])}</strong><code>{esc(item["code"])}</code><p>{esc(item["text"])}</p></div>'
        nodes = self.content["architecture_nodes"]
        text = ''.join(node(item) for item in nodes[:2])
        fields = ''.join(node(item) for item in nodes[2:4])
        merge = node(nodes[4])
        return f'<figure class="architecture-figure"><div class="architecture-grid"><div class="architecture-branch"><h3 class="branch-label">{self.t("text_branch")}</h3>{text}</div><div class="architecture-branch"><h3 class="branch-label">{self.t("comparison_branch")}</h3>{fields}</div><div class="architecture-fusion">{merge}</div><div class="architecture-output"><strong>{self.t("architecture_output")}</strong></div></div></figure>'

    def aggregation(self) -> str:
        formula = '<div class="aggregation-formulas"><p><code>pₛ = sigmoid((zₛ,AB + zₛ,BA) / 2)</code></p><p><code>p = mean(p₄₂, p₄₃, p₄₄)</code></p></div>'
        return self.p("aggregation_p", "section-intro") + formula + self.p("aggregation_note", "caption")

    def method(self) -> str:
        role_keys = ("fit", "dev", "calibration", "evaluation")
        roles = ''.join(f'<article class="role-item"><h3>{esc(self.content["role_names"][i])}</h3><p class="role-count" data-source="roles.{key}">{self.results["roles"][key]:,}</p><p>{esc(self.content["role_p"][i])}</p></article>' for i, key in enumerate(role_keys))
        roles = self.p("data_context", "section-intro") + self.p("roles_intro") + f'<div class="role-list">{roles}</div>' + self.p("roles_note", "caption")
        roles += self.link(self.doc("DATA_CARD"), self.content["data_card_link"])
        processing = self.p("processing_intro", "section-intro") + '<ul class="processing-list">' + ''.join(f'<li><strong>{esc(title)}</strong><p>{esc(self.content["processing_p"][i])}</p></li>' for i, title in enumerate(self.content["processing_names"])) + '</ul>' + self.normalization_examples()
        features, offset = [], 0
        for i, group in enumerate(self.features["groups"]):
            entries = []
            for index in range(offset, offset + group["count"]):
                name, label = self.features["names"][index], self.content["feature_labels"][index]
                entries.append(f'<li data-feature-index="{index}"><code>{esc(name)}</code><span class="feature-meaning">{esc(label)}</span></li>')
            features.append(f'<details id="feature-group-{i+1}"><summary>{esc(self.content["feature_group_names"][i])}<span class="feature-count">{group["count"]} {self.t("feature_count")}</span></summary><p class="feature-note">{esc(self.content["feature_group_p"][i])}</p><ul>{"".join(entries)}</ul></details>')
            offset += group["count"]
        feature_body = self.p("features_intro", "section-intro") + f'<div class="feature-groups">{"".join(features)}</div>' + f'<p class="caption">{self.link("../../reproducibility/FEATURES.json",self.content["feature_source"],download=True)}</p>'
        architecture = self.p("architecture_intro", "section-intro") + self.diagram()
        architecture += self.details("architecture_budget_title", self.p("architecture_budget_p"), "token-budgets")
        config = self.p("configuration_intro", "section-intro") + self.p("calibration_p")
        config += self.details("training_settings_title", self.configuration_table(), "training-settings")
        config += f'<p class="caption">{self.link("results.html#limitations", self.content["read_scope"])}</p>'
        return self.section("roles", "roles_title", roles, "01", True) + self.section("processing", "processing_title", processing, "02") + self.section("features", "features_title", feature_body, "03") + self.section("architecture", "architecture_title", architecture, "04", True) + self.section("aggregation", "model_title", self.aggregation(), "05") + self.section("configuration", "configuration_title", config, "06")

    def configuration_table(self) -> str:
        m = self.model
        configs = [m["model_revision"], ", ".join(map(str, m["seeds"])), m["selected_epoch"], f'{m["micro_batch"]} / {m["effective_batch"]}', f'{m["encoder_lr"]} / {m["head_lr"]}', f'{m["scheduler_epochs"]} / {m["warmup_fraction"]}', f'{m["weight_decay"]} / {m["gradient_clip"]}', m["dropout"], self.content["precision_training"], self.content["precision_inference"]]
        rows = ''.join(f'<tr><th scope="row" class="config-key">{esc(label)}</th><td class="config-value">{esc(value)}</td></tr>' for label, value in zip(self.content["config_labels"], configs))
        return '<div class="table-wrap config-table"><table class="table"><tbody>' + rows + '</tbody></table></div>'

    def method_resources(self) -> str:
        """Readable inline resources; all values are rendered locally, with no fetch."""
        m = self.model
        overview = ''.join(f'<article><h3>{esc(item["title"])}</h3><p>{esc(item["text"])}</p></article>' for item in self.content["model_resource_blocks"])
        model_body = self.p("model_definition") + f'<div class="resource-summary-grid">{overview}</div>'
        model_body += f'<div class="resource-reading-links">{self.link("#architecture", self.content["view_architecture"])}{self.link("#features", self.content["view_features"])}</div>'
        model_body += f'<p class="caption">{self.link("results.html#limitations", self.content["read_scope"])}</p>'
        config_body = self.p("configuration_intro") + self.configuration_table()
        config_body += f'<p class="resource-cutoff"><strong>{self.tr("校准阈值", "Calibration cutoff")}</strong><code>{self.context["threshold"]}</code></p>'
        checkpoints = ''.join(f'<li><strong>{self.tr("种子", "Seed")} {item["seed"]}</strong><span>{item["bytes"]:,} bytes</span><code>{esc(item["sha256"])}</code></li>' for item in m["checkpoints"])
        config_body += f'<details class="checkpoint-details"><summary>{self.t("checkpoint_title")}</summary>{self.p("checkpoint_note")}<ul>{checkpoints}</ul></details>'
        bodies = (("model", model_body), ("config", config_body))
        panels = ''.join(f'<section id="{key}-panel" class="resource-panel" data-resource-panel="{key}" aria-labelledby="{key}-panel-title"><h2 id="{key}-panel-title">{esc(self.content["pages"]["method"]["links"][i])}</h2>{body}</section>' for i, (key, body) in enumerate(bodies))
        return f'<div class="resource-panels">{panels}</div>'

    def plot(self, metric: str) -> str:
        rows = []
        for name in METHODS:
            value = self.results["metrics"][name][metric]
            css = " is-final" if name == "C+" else " is-ablation" if name == "C0" else ""
            rows.append(f'<div class="chart-row{css}" data-method="{name}" data-value="{value}"><span class="chart-method">{name}</span><span class="chart-track" aria-hidden="true"><span class="chart-bar" style="width:{value*100:.8f}%"></span></span><span class="chart-value">{value:.4f}</span></div>')
        title_key = "chart_ap_title" if metric == "ap" else "chart_f1_title"
        caption_key = "chart_ap_caption" if metric == "ap" else "chart_f1_caption"
        return f'<figure class="metric-figure" id="plot-{metric}" data-metric-plot="{metric}" data-axis-min="0" data-axis-max="1" aria-labelledby="plot-title-{metric}"><h3 id="plot-title-{metric}">{self.t(title_key)}</h3><div class="chart-axis" aria-hidden="true"><span>0</span><span>0.25</span><span>0.50</span><span>0.75</span><span>1.00</span></div>{"".join(rows)}<figcaption class="chart-foot">{self.t(caption_key)}</figcaption></figure>'

    def results_page(self) -> str:
        controls = f'<div class="metric-toolbar"><div class="metric-options" role="group" aria-label="{self.t("metric_control")}"><a class="metric-button" role="button" href="?metric=ap#comparison" data-metric="ap" aria-pressed="false">AP</a><a class="metric-button" role="button" href="?metric=f1#comparison" data-metric="f1" aria-pressed="false">F1</a></div><p>{self.t("chart_range")}</p></div>'
        references = ''.join(f'<div><dt>{name}</dt><dd>{esc(self.content["reference_methods"][name])}</dd></div>' for name in METHODS)
        definitions = f'<div class="comparison-guide"><h3>{self.t("reference_title")}</h3><dl class="reference-list">{references}</dl>{self.p("reference_control_note")}</div>'
        plots = self.p("comparison_intro", "section-intro") + definitions + controls + self.plot("ap") + self.plot("f1") + f'<noscript><p class="noscript-note">{self.t("nojs")}</p></noscript>'
        table_rows = []
        keys = ("ap", "f1", "p_at_100", "precision", "recall")
        for name in METHODS:
            m = self.results["metrics"][name]
            cells = ''.join(f'<td class="numeric" data-metric="{key}" data-value="{m[key]}">{m[key]:.2f}</td>' if key == "p_at_100" else f'<td class="numeric" data-metric="{key}" data-value="{m[key]}">{m[key]:.6f}</td>' for key in keys)
            row_class = ' class="final-row"' if name == "C+" else ""
            table_rows.append(f'<tr data-method="{name}"{row_class}><th scope="row">{name}<span class="method-description">{esc(self.content["methods"][name])}</span></th>{cells}</tr>')
        heads = f'<th scope="col">{self.t("method_column")}</th><th scope="col" class="numeric">AP</th><th scope="col" class="numeric">F1</th><th scope="col" class="numeric">P@100</th><th scope="col" class="numeric">{self.t("precision_label")}</th><th scope="col" class="numeric">{self.t("recall_label")}</th>'
        table = f'<div class="table-wrap"><table class="table" id="results-table"><thead><tr>{heads}</tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>' + self.p("table_caption", "caption")
        cm = self.results["metrics"]["C+"]["confusion"]
        def cell(key: str, error: bool) -> str:
            cell_class = ' class="error"' if error else ""
            return f'<td data-confusion="{key}"{cell_class}><span class="matrix-code">{key.upper()}</span><span class="matrix-value">{cm[key]}</span></td>'
        matrix = f'<figure><table class="confusion-matrix" aria-label="{self.t("matrix_title")}"><thead><tr><th scope="col">{self.t("actual_label")}</th><th scope="col">{self.t("predicted_negative")}</th><th scope="col">{self.t("predicted_positive")}</th></tr></thead><tbody><tr><th scope="row">{self.t("actual_negative")}</th>{cell("tn",False)}{cell("fp",True)}</tr><tr><th scope="row">{self.t("actual_positive")}</th>{cell("fn",True)}{cell("tp",False)}</tr></tbody></table><figcaption class="caption">{self.t("matrix_caption")}</figcaption></figure>'
        matrix = f'<div class="matrix-and-note">{matrix}<div class="prose"><h3>{self.t("matrix_note_title")}</h3>{self.p("matrix_note")}</div></div>'
        definitions = '<div class="metric-definition-list">' + ''.join(f'<article><h3>{label}</h3><p>{esc(text)}</p></article>' for label, text in zip(("AP", "F1", "P@100"), self.content["definition_p"])) + '</div>'
        limits = '<div class="prose">' + ''.join(f'<p>{esc(paragraph)}</p>' for paragraph in self.content["limits_paragraphs"]) + self.link(self.doc("LIMITATIONS"),self.content["limitations_doc_link"]) + '</div>'
        return self.section("comparison", "comparison_title", plots, "01") + self.section("gains", "delta_title", self.gains(), "02") + self.section("full-table", "table_title", table, "03") + self.section("confusion", "matrix_title", matrix + self.outcomes(), "04", True) + self.section("metrics", "definitions_title", definitions, "05") + self.section("limitations", "limits_title", limits, "06", True)

    def gains(self) -> str:
        rows = []
        for reference in ("L42", "C0"):
            for key in ("ap", "f1"):
                delta = 100 * (self.results['metrics']['C+'][key] - self.results['metrics'][reference][key])
                label = f'{key.upper()} · C+ − {reference}'
                rows.append(f'<div class="delta-row" data-delta-reference="{reference}" data-delta-metric="{key}" data-value="{delta}"><span>{label}</span><span class="delta-track" aria-hidden="true"><span class="delta-bar" style="width:{delta/6*100:.8f}%"></span></span><strong>+{delta:.4f} {self.tr("百分点","pp")}</strong></div>')
        axes = '<div class="chart-axis delta-axis"><span>0</span><span>2</span><span>4</span><span>6</span></div>'
        return self.p("gains_intro", "section-intro") + f'<figure class="delta-chart" data-axis-min="0" data-axis-max="6">{axes}{"".join(rows)}<figcaption class="caption">{self.t("gains_caption")}</figcaption></figure>'

    def outcomes(self) -> str:
        cm = self.results['metrics']['C+']['confusion']
        rows = []
        for group, correct, error in [('positive','tp','fn'),('negative','tn','fp')]:
            total = cm[correct] + cm[error]
            name = self.t("outcome_" + group)
            error_name = self.t("missed_matches" if group == "positive" else "false_matches")
            rows.append(f'<div class="outcome-row" data-outcome="{group}"><h3>{name} · {total}</h3><div class="outcome-bar" aria-hidden="true"><span class="outcome-correct" style="width:{100*cm[correct]/total:.8f}%"></span><span class="outcome-error" style="width:{100*cm[error]/total:.8f}%"></span></div><div class="outcome-labels"><span>{self.t("correct")}: {cm[correct]} ({100*cm[correct]/total:.2f}%)</span><strong>{error_name}: {cm[error]} ({100*cm[error]/total:.2f}%)</strong></div></div>')
        return f'<div class="outcome-chart"><h3>{self.t("outcomes_title")}</h3>{self.p("outcomes_intro")}{"".join(rows)}{self.p("outcomes_tradeoff", "caption")}</div>'

    def command_flow(self) -> str:
        body = ''.join(f'<tr><th scope="row"><code>{esc(stage["name"])}</code></th><td>{esc(stage["input"])}</td><td>{esc(stage["operation"])}</td><td>{esc(stage["output"])}</td></tr>' for stage in self.content["command_stages"])
        heads = ''.join(f'<th scope="col">{esc(label)}</th>' for label in self.content["command_columns"])
        return self.p("command_flow_intro", "section-intro") + f'<div class="table-wrap"><table><thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table></div>{self.link(self.doc("CORE_USAGE"),self.content["pages"]["reproduce"]["links"][1])}'

    def toc(self, page: str) -> str:
        keys = {
            'index':[('project','brief_title'),('example','overview_example_title'),('evidence','overview_choice_title'),('summary','overview_result_title'),('public-code','overview_close_title')],
            'method':[('roles','roles_title'),('processing','processing_title'),('features','features_title'),('architecture','architecture_title'),('aggregation','model_title'),('configuration','configuration_title')],
            'results':[('comparison','comparison_title'),('gains','delta_title'),('full-table','table_title'),('confusion','matrix_title'),('metrics','definitions_title'),('limitations','limits_title')],
            'reproduce':[('quickstart','quickstart_title'),('public-checks','public_title'),('command-flow','command_flow_title'),('replay','replay_title'),('fresh-training','fresh_title'),('resources','resources_title'),('rebuild','rebuild_title')],
        }[page]
        links = ''.join(f'<a href="#{anchor}"><span>{i+1:02}</span>{self.t(key)}</a>' for i,(anchor,key) in enumerate(keys))
        return f'<p class="toc-label">{self.tr("本页内容","On this page")}</p><nav aria-label="{self.tr("页内目录","Page contents")}">{links}</nav>'

    def command(self, identity: str, lines: list[str]) -> str:
        text = "\n".join(lines)
        return f'<div class="command-block"><pre id="{identity}"><code>{esc(text)}</code></pre><button class="copy-button" type="button" data-copy-target="{identity}" data-copy-status="{identity}-status" hidden>{self.t("copy")}</button></div><span class="copy-status" id="{identity}-status" role="status" aria-live="polite"></span>'

    def python_commands(self, identity: str, arguments: list[str], heading: int = 3) -> str:
        unix = [".venv/bin/python " + line for line in arguments]
        windows = [r".\.venv\Scripts\python.exe " + line for line in arguments]
        return f'<h{heading}>macOS / Linux</h{heading}>' + self.command(identity+"-unix", unix) + f'<h{heading}>Windows · PowerShell</h{heading}>' + self.command(identity+"-windows", windows)

    def reproduce(self) -> str:
        setup = self.p("quickstart_intro", "section-intro")
        setup += self.command("clone-command", ["git clone https://github.com/Yemyu/product-entity-matching.git", "cd product-entity-matching"])
        setup += '<h3>macOS / Linux</h3>' + self.command("setup-unix", ["python3 -m venv .venv", ".venv/bin/python scripts/public_smoke.py", ".venv/bin/python -m http.server 8000 --bind 127.0.0.1"])
        setup += '<h3>Windows · PowerShell</h3>' + self.command("setup-windows", ["py -3.11 -m venv .venv", r".\.venv\Scripts\python.exe scripts/public_smoke.py", r".\.venv\Scripts\python.exe -m http.server 8000 --bind 127.0.0.1"])
        setup += self.p("quickstart_preview") + self.link("http://127.0.0.1:8000/showcase/"+self.language+"/index.html", self.content["quickstart_open"])
        setup += self.p("quickstart_python", "caption")
        setup += self.p("quickstart_gpu") + self.link(self.doc("REPRODUCIBILITY"),self.content["quickstart_gpu_link"])
        steps = []
        for i, (title, text, commands) in enumerate([
            ("check_step_title", "check_step_p", ["scripts/public_smoke.py", "scripts/run_notebook.py", "scripts/verify_public_release.py"]),
            ("tests_step_title", "tests_step_p", ["scripts/matching.py --help", "scripts/run_tests.py"]),
        ], 1):
            steps.append(f'<article class="reproduction-step"><div class="step-index" aria-hidden="true">0{i}</div><div><h3>{self.t(title)}</h3>{self.p(text)}{self.python_commands("public-command-"+str(i),commands,4)}</div></article>')
        public = self.p("public_intro", "section-intro") + "".join(steps)
        replay = self.p("replay_intro", "section-intro") + '<ul class="replay-list">' + ''.join(f'<li>{esc(item)}</li>' for item in self.content["replay_items"]) + '</ul>' + self.p("replay_note", "caption")
        fresh = f'<div class="prose">{self.p("fresh_p")}{self.p("fresh_note")}</div>'
        resource_urls = [self.doc(name) for name in DOCS] + ["../../results/fixed-results.json", "../../reproducibility/FINAL_MODEL.json", "../../reproducibility/FEATURES.json"]
        resources = '<div class="resource-list">' + ''.join(self.link(url, label, "resource-link", download=url.endswith(".json")) for url, label in zip(resource_urls, self.content["resource_names"])) + '</div>'
        resources += f'<p class="caption">{self.link("https://github.com/Yemyu/product-entity-matching/tree/main/src/product_matching",self.content["source_project"])}</p>'
        rebuild = self.p("rebuild_p", "section-intro") + self.python_commands("build-command", ["scripts/build_showcase.py", "scripts/build_showcase.py --check"]) + self.p("rebuild_check_p", "caption")
        return self.section("quickstart", "quickstart_title", setup, "01") + self.section("public-checks", "public_title", public, "02") + self.section("command-flow", "command_flow_title", self.command_flow(), "03") + self.section("replay", "replay_title", replay, "04") + self.section("fresh-training", "fresh_title", fresh, "05", True) + self.section("resources", "resources_title", resources, "06") + self.section("rebuild", "rebuild_title", rebuild, "07")

    def render(self, page: str, template: str) -> bytes:
        c, entry = self.content, self.content["pages"][page]
        index = PAGES.index(page)
        navigation = []
        for i, name in enumerate(PAGES):
            current = ' aria-current="page"' if name == page else ""
            navigation.append(f'<a class="nav-link" href="{name}.html"{current}>{esc(c["nav"][i])}</a>')
        navigation = ''.join(navigation)
        urls = {
            "index": ("results.html#comparison", "method.html"),
            "method": (self.doc("MODEL_CARD"), "../../reproducibility/FINAL_MODEL.json"),
            "results": ("../assets/fixed-results.csv", "#limitations"),
            "reproduce": ("#quickstart", self.doc("CORE_USAGE")),
        }[page]
        hero_links = ''.join(self.link(url, entry["links"][i], "external-link button is-normal " + ("is-dark" if i == 0 else "is-light"), page == "results" and i == 0) for i, url in enumerate(urls))
        if page == "method":
            hero_links = ''.join(f'<a id="{key}-toggle" href="#{key}-panel" role="button" class="button resource-toggle {"is-dark" if i == 0 else "is-light"}" data-resource-toggle="{key}" aria-controls="{key}-panel" aria-expanded="true">{esc(entry["links"][i])}<span class="resource-chevron" aria-hidden="true">⌄</span></a>' for i, key in enumerate(("model", "config")))
        next_page = PAGES[(index+1) % len(PAGES)]
        next_link = f'<a class="next-link" href="{next_page}.html"><span><span class="next-eyebrow">{self.t("next")}</span>{esc(c["pages"][next_page]["title"])}</span><span aria-hidden="true">→</span></a>'
        body = {"index": self.overview, "method": self.method, "results": self.results_page, "reproduce": self.reproduce}[page]()
        mapping = {"LANG": self.language, "DESCRIPTION": esc(self.fill(entry["description"])), "PAGE_TITLE": esc(entry["title"]), "PROJECT_NAME": self.t("project"), "PAGE_FILE": page+".html", "PAGE_KEY": page, "SKIP": self.t("skip"), "BRAND": self.t("brand"), "NAV_LABEL": self.t("nav_label"), "NAVIGATION": navigation, "LANGUAGE_URL": "../"+self.other+"/"+page+".html", "OTHER_LANG": self.other, "LANGUAGE_LABEL": self.t("language_label"), "LANGUAGE_TEXT": self.t("language_text"), "EYEBROW": esc(entry["eyebrow"]), "HEADING": esc(self.fill(entry["heading"])), "LEAD": esc(self.fill(entry["lead"])), "HERO_LINKS": hero_links, "HERO_FACTS": self.hero_facts() if page=="index" else "", "BODY": body, "NEXT_LINK": next_link, "FOOTER_SCOPE": self.t("footer_scope"), "CREDIT": c["credit"], "LICENSE_SCOPE": c["license_scope"], "NOTICE_LABEL": self.t("notice"), "PAGE_TOC": self.toc(page)}
        mapping["THEME_COLOR"] = "#f5efe4"
        mapping["MODEL_DEFINITION"] = self.t("model_definition")
        mapping["NOTICE_URL"] = GITHUB + "/blob/main/showcase/NOTICE.md"
        mapping["HERO_PANELS"] = self.method_resources() if page == "method" else ""
        mapping["ASSET_VERSION"] = hashlib.sha256((SITE / "assets/site.js").read_bytes() + (SITE / "assets/site.css").read_bytes()).hexdigest()[:12]
        output = re.sub(r"\{\{([A-Z_]+)\}\}", lambda match: mapping[match[1]], template)
        assert not re.search(r"\{\{[A-Z_]+\}\}|\[\[[a-zA-Z0-9_]+\]\]", output), "Unresolved template field"
        return ("\n".join(line.rstrip() for line in output.splitlines()) + "\n").encode("utf-8")


def build_outputs() -> dict[Path, bytes]:
    results = read_json(ROOT / "results/fixed-results.json")
    model = read_json(ROOT / "reproducibility/FINAL_MODEL.json")
    features = read_json(ROOT / "reproducibility/FEATURES.json")
    assert results["model"]["lexical_features"] == model["feature_count"] == len(features["names"])
    assert results["roles"] == model["roles"]
    template = (SITE / "templates/page.html").read_text(encoding="utf-8")
    outputs = {}
    for language in LANGUAGES:
        content = read_json(SITE / "content" / (language + ".json"))
        assert len(content["feature_labels"]) == len(features["names"])
        renderer = Renderer(language, content, results, model, features)
        for page in PAGES:
            outputs[SITE / language / (page + ".html")] = renderer.render(page, template)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    fields = ("ap", "f1", "p_at_100", "precision", "recall", "threshold", "count", "positive_count")
    writer.writerow(("method",) + fields + ("tn", "fp", "fn", "tp"))
    for method in METHODS:
        row = results["metrics"][method]
        writer.writerow([method] + [row[field] for field in fields] + [row["confusion"][key] for key in ("tn", "fp", "fn", "tp")])
    outputs[SITE / "assets/fixed-results.csv"] = stream.getvalue().encode("utf-8")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare generated bytes in memory; do not write files.")
    args = parser.parse_args()
    outputs = build_outputs()
    mismatches = [str(path.relative_to(ROOT)) for path, data in outputs.items() if not path.is_file() or path.read_bytes() != data]
    if args.check:
        report = {"status": "passed" if not mismatches else "failed", "mode": "check", "writes": False, "generated_files": len(outputs), "mismatches": mismatches}
        print(json.dumps(report, ensure_ascii=False))
        return int(bool(mismatches))
    for path, data in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(json.dumps({"status": "built", "html_pages": 8, "generated_files": len(outputs), "network": False, "model_execution": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
