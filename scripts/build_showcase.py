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

    def stage(self, index: int) -> str:
        stage = self.content["walkthrough_stages"][index]
        labels = [("input", "输入", "Input"), ("operation", "处理", "Operation"),
                  ("output", "输出", "Output"), ("rationale", "选择依据", "Reason for the choice")]
        parts = ''.join(f'<div><h4>{self.tr(zh,en)}</h4><p>{esc(stage[key])}</p></div>' for key, zh, en in labels)
        return f'<article class="stage-card" aria-label="{esc(stage["title"])}"><div class="stage-grid">{parts}</div></article>'

    def workflow(self) -> str:
        steps = ''.join(f'<li><span>{i+1:02}</span><strong>{esc(s["title"])}</strong></li>'
                        for i, s in enumerate(self.content["walkthrough_stages"]))
        return f'<ol class="workflow-strip">{steps}</ol>'

    def normalization_examples(self) -> str:
        rows = [("text", "ＡＣＭＥ  ZX-100", normalize_text("ＡＣＭＥ  ZX-100"), "present")]
        for raw in ("1.999e1", "$19.99", "19,99", None):
            result = parse_price(raw)
            rows.append(("price",raw,result.normalized,result.status))
        body = ''.join(f'<tr><th scope="row"><code>{field}</code></th><td><code>{esc(raw) if raw is not None else "null"}</code></td><td><code>{esc(normalized) if normalized is not None else "null"}</code></td><td><code>{state}</code></td></tr>' for field,raw,normalized,state in rows)
        heads = ''.join(f'<th scope="col">{self.tr(z,e)}</th>' for z,e in [('字段','Field'),('原值','Raw value'),('比较视图','Comparison view'),('质量状态','Quality state')])
        note = self.tr("这些教学值直接调用公开规范化函数。普通文本保留连字符；只在另外的紧凑型号/标题比较视图中移除分隔。价格包含货币符号或逗号时不猜测其地区格式，也不把缺失填成0。", "These teaching values call the public normalization functions. Ordinary text retains hyphens; separators are removed only in separate compact model/title views. Currency symbols and commas remain ambiguous rather than guessing a locale; missing prices are not filled with zero.")
        return f'<div class="table-wrap"><table><thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table></div><p class="caption">{note}</p>'

    def workbench(self) -> str:
        """Run only pure feature calculations on invented records; no model/IDF fitting."""
        cases = [
            ("spelling", "型号书写差异", "Model spelling", "ZX100", "ACME", "89.00", "USD"),
            ("digits", "型号数字差异", "Different model digits", "ZX101", "ACME", "89.00", "USD"),
            ("missing", "品牌缺失与币种不同", "Missing brand / different currency", "ZX100", None, "89.00", "EUR"),
        ]
        labels = [
            ("title_token_jaccard", "标题词语重合", "Title token overlap"),
            ("brand_equal", "品牌相同且均存在", "Equal, present brands"),
            ("brand_missing_any", "至少一个品牌缺失", "Any brand missing"),
            ("price_comparable", "价格可比较", "Comparable prices"),
            ("price_similarity", "同币种价格相似度", "Same-currency price similarity"),
            ("code_exact_overlap", "标题型号代码重合", "Title model-code overlap"),
            ("code_same_letters_different_digits", "型号字母相同、数字不同", "Same model letters, different digits"),
            ("native_model_compact_exact", "紧凑型号相同且均存在", "Equal, present compact model numbers"),
        ]
        tabs, panels = [], []
        for identity, zh, en, model, brand, price, currency in cases:
            left = {"title":"Acme ZX-100 cordless drill","brand":"Acme","price":"89.00","priceCurrency":"USD"}
            right = {"title":f"ACME {model} cordless drill","brand":brand,"price":price,"priceCurrency":currency}
            a, b = business_record(left), business_record(right)
            an = native_record({"modelno":"ZX-100","category":"power tools"})
            bn = native_record({"modelno":model,"category":"power tools"})
            values = {**pair_features(a,b), **local_code_features(a['normalized']['title'],b['normalized']['title']),
                      **native_features(an,bn,a['normalized']['title'],b['normalized']['title'])}
            tabs.append(f'<button type="button" data-example-case="{identity}" aria-pressed="false">{self.tr(zh,en)}</button>')
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
            rows = ''.join(f'<tr data-demo-feature="{name}" data-demo-value="{values[name]}"><th scope="row">{self.tr(z,e)}<code>{name}</code></th><td>{values[name]:.4f}</td></tr>' for name,z,e in labels)
            features = f'<div class="table-wrap field-comparison"><table><thead><tr><th>{self.tr("选取的比较特征","Selected comparison features")}</th><th>{self.tr("程序计算值","Computed value")}</th></tr></thead><tbody>{rows}</tbody></table></div>'
            notes = {
                "spelling":("标题token重合只有0.5，因为ZX-100与ZX100被分成了不同token。紧凑型号比较和代码重合均为1，保留了书写格式之外的一致性。", "Title token overlap is only 0.5: ZX-100 and ZX100 tokenize differently. Compact-model equality and code overlap are both 1, retaining evidence beyond spelling."),
                "digits":("标题token重合仍为0.5，但紧凑型号比较为0，数字冲突特征为1。程序将这些证据交给模型，不使用一条型号规则强行判定匹配或不匹配。", "Title token overlap remains 0.5, but compact-model equality is 0 and the digit-conflict feature is 1. These values are model inputs, not a hard match or rejection rule."),
                "missing":("品牌缺失不会算作品牌相同；两笔价格都可解析，但USD与EUR不同，价格可比较性与价格相似度均为0。这里的0表示不可比较，必须结合可比较性指示一起解释。", "A missing brand does not count as an equal brand. Both prices parse, but USD and EUR differ, so price comparability and similarity are 0. Here zero means unavailable comparison and must be read with its presence flag."),
            }
            panels.append(f'<article id="example-{identity}" data-example-panel="{identity}" aria-labelledby="example-title-{identity}"><h3 id="example-title-{identity}">{self.tr(zh,en)}</h3>{fields}{features}<p class="example-explanation">{self.tr(*notes[identity])}</p></article>')
        note = self.tr("以下商品为虚构教学输入。展示值由公开特征函数实际计算；没有调用RoBERTa，也没有匹配标签、模型概率或身份结论。", "The listings below are invented teaching inputs. Values are computed by the public feature functions; no RoBERTa call, match label, model probability or identity conclusion is supplied.")
        return f'<div class="field-workbench"><p class="lead-note">{note}</p><div class="example-controls" role="group" aria-label="{self.tr("比较场景","Comparison scenarios")}">{"".join(tabs)}</div>{"".join(panels)}</div>'

    def p(self, key: str, css: str = "") -> str:
        attribute = f' class="{css}"' if css else ""
        return f'<p{attribute}>{self.t(key)}</p>'

    def doc(self, name: str) -> str:
        directory = "../../docs/" + ("zh-CN/" if self.language == "zh-CN" else "")
        return directory + name + ".md"

    def section(self, anchor: str, title: str, body: str, number: str, tint: bool = False) -> str:
        css = "section section-rule" + (" section-tint" if tint else "")
        return f'<section id="{anchor}" class="{css}"><div class="container"><div class="section-heading"><span class="section-number" aria-hidden="true">{number}</span><h2 class="title is-3">{self.t(title)}</h2></div>{body}</div></section>'

    def link(self, url: str, text: str, css: str = "text-link", download: bool = False) -> str:
        return f'<a href="{esc(url)}" class="{css}"{ " download" if download else ""}>{esc(text)}</a>'

    def overview(self) -> str:
        blocks = [
            ("商品匹配", "Product matching", "同一商品在不同零售商处可能使用不同标题、型号写法或不完整的字段。识别这些对应关系，可以帮助对齐商品目录和检查重复记录。", "The same product can appear under different titles, model-number spellings or incomplete fields at different retailers. Identifying these correspondences helps align catalogues and check duplicate records."),
            ("输入与输出", "Input and output", "输入是一对商品记录，包含标题、品牌、描述、价格、币种及原生型号和类别。模型输出0到1的匹配分数；将分数与校准阈值比较后，得到匹配或不匹配的判断。", "The input is a pair of listings with title, brand, description, price, currency, and native model/category fields. The model outputs a match score between 0 and 1; comparing it with the calibration cutoff gives a match or non-match decision."),
            ("数据与输入", "Data and inputs", "Walmart–Amazon商品匹配数据。每对保留业务字段及原生型号/类别，用四种固定用途分开处理模型拟合、选轮数、校准和评价。", "Walmart–Amazon product matching data. Business fields and native model/category fields are retained, with separate roles for fitting, epoch selection, calibration and evaluation."),
            ("实现内容", "Implementation", "构造42项比较特征，与RoBERTa的商品对文本表示融合；比较四种参考方法与一个独立训练的零特征对照。", "Construct 42 comparison features and fuse them with RoBERTa pair representations; compare four reference methods and a separately trained zero-feature control."),
        ]
        brief = '<div class="project-brief"><div class="brief-grid">' + ''.join(f'<article class="brief-block"><h3>{self.tr(z,e)}</h3><p>{self.tr(zp,ep)}</p></article>' for z,e,zp,ep in blocks) + '</div></div>'
        pipeline = self.workflow() + self.p("model_decision_explanation", "section-intro")
        example = self.p("data_example_intro", "section-intro") + self.workbench()
        metrics = self.results["metrics"]["C+"]
        summaries = []
        for key, label, context in (("ap", "AP", "ap"), ("f1", "F1", "f1"), ("p_at_100", "P@100", "p100")):
            value = metrics[key]
            formatted = f"{value:.2f}" if key == "p_at_100" else f"{value:.4f}"
            summaries.append(f'<div class="summary-metric"><p class="summary-label"><abbr title="{self.t(context+"_expansion")}">{label}</abbr></p><p class="summary-value" data-source="metrics.C+.{key}" data-value="{value}">{formatted}</p><p class="summary-context">{self.t(context+"_context")}</p></div>')
        result = f'<p class="note-kicker">{self.t("evaluation_context")}</p><div class="results-summary">{"".join(summaries)}</div>{self.p("result_interpretation", "section-intro")}{self.p("scope", "lead-note")}{self.link("results.html",self.content["read_results"])}'
        close = f'<div class="prose">{self.p("overview_close_p")}{self.link("reproduce.html#quickstart",self.content["read_reproduce"])}</div>'
        return self.section("project", "brief_title", brief, "01") + self.section("evidence", "overview_choice_title", pipeline, "02") + self.section("example", "overview_example_title", example, "03") + self.section("summary", "overview_result_title", result, "04") + self.section("public-code", "overview_close_title", close, "05")

    def diagram(self) -> str:
        def node(zh, en, code, zp, ep):
            return f'<div class="architecture-node"><strong>{self.tr(zh,en)}</strong><code>{esc(code)}</code><p>{self.tr(zp,ep)}</p></div>'
        text = node("字段序列化", "Field serialization", "brand 16 · model 32 · category 16 · title 62", "两条记录以AB与BA两种顺序组成完整商品对。每条记录最多126个token，加4个特殊token后商品对最多256个。描述和价格不进入此文本序列。", "Serialize the pair in both AB and BA order. Each record uses at most 126 tokens; four special tokens bring the pair limit to 256. Description and price are not in this text sequence.")
        text += node("商品对联合编码", "Joint pair encoding", "RoBERTa-base · CLS 768 → 64", "编码器联合读取整个商品对，CLS经线性层和GELU投影为64维。并不是分别编码商品再计算向量余弦。", "RoBERTa jointly reads the complete pair. Its CLS state is projected through a linear layer and GELU to 64 dimensions; the model does not score two independent embeddings by cosine similarity.")
        fields = node("显式比较", "Explicit comparisons", "14 + 8 + 8 + 12 = 42", "业务字段、标题证据、标题型号代码、原生型号与类别。缺失/可比较性单独编码；IDF只在训练数据上拟合。", "Business fields, title evidence, title model codes, native model/category comparisons. Presence and comparability have separate flags; IDF is fitted on training data only.")
        fields += node("固定顺序向量", "Fixed-order vector", "x42 ∈ [0, 1]⁴²", "相同商品对的特征用于两个输入方向。描述、价格及币种通过这条分支参与比较。", "The same feature vector is used in both directions. Description, price and currency contribute through this branch.")
        merge = node("连接与分类头", "Concatenation and classifier", "64 + 42 = 106 → 64 → 1", "连接文本投影与42项特征，经64维GELU层和0.1 dropout输出一个logit。训练更新编码器与分类头。", "Concatenate the text projection and 42 features. A 64-unit GELU layer with 0.1 dropout produces one logit. Training updates both encoder and classifier.")
        return f'<figure class="architecture-figure"><div class="architecture-grid"><div class="architecture-branch"><h3 class="branch-label">{self.tr("文本分支","Text branch")}</h3>{text}</div><div class="architecture-branch"><h3 class="branch-label">{self.tr("比较特征分支","Comparison-feature branch")}</h3>{fields}</div><div class="architecture-fusion">{merge}</div><div class="architecture-output"><strong>{self.tr("每个方向输出未校准logit，随后进行方向与种子聚合。","Each direction yields a logit, followed by direction and seed aggregation.")}</strong></div></div><figcaption class="caption">{self.t("architecture_note")}</figcaption></figure>'

    def aggregation(self) -> str:
        explanation = self.tr("在每个种子内，先求AB与BA的logit均值，再用sigmoid映射成概率。之后对42、43、44三个种子的概率等权平均，得到最终分数。两个操作的顺序固定；不先平均所有logit，也不选择分数最好的一个种子。", "Within each seed, average the AB and BA logits, then apply sigmoid. Average the resulting probabilities across seeds 42, 43 and 44. The order is fixed: do not average all logits together or choose one seed by its score.")
        formula = '<div class="aggregation-formulas"><p><code>pₛ = sigmoid((zₛ,AB + zₛ,BA) / 2)</code></p><p><code>p = (p₄₂ + p₄₃ + p₄₄) / 3</code></p></div>'
        note = self.tr("训练按6轮调度运行，开发数据为三个种子选择共同第4轮检查点。集成分数随后用校准集选定的阈值0.3947772259513537进行判定：分数大于或等于阈值为匹配。阈值不是0.5，也不在评价数据上重选。", "Training follows the six-epoch schedule; development data selects the shared epoch-4 checkpoints. The ensemble score is classified using the calibration cutoff 0.3947772259513537: scores at or above it are matches. The cutoff is not 0.5 and is not reselected on evaluation data.")
        return self.stage(4) + f'<p class="section-intro">{explanation}</p>{formula}<p class="caption">{note}</p>'

    def method(self) -> str:
        role_keys = ("fit", "dev", "calibration", "evaluation")
        roles = ''.join(f'<article class="role-item"><h3>{esc(self.content["role_names"][i])}</h3><p class="role-count" data-source="roles.{key}">{self.results["roles"][key]:,}</p><p>{esc(self.content["role_p"][i])}</p></article>' for i, key in enumerate(role_keys))
        roles = self.stage(1) + self.p("roles_intro", "section-intro") + f'<div class="role-list">{roles}</div>' + self.p("roles_note", "caption")
        processing = self.stage(0) + '<ul class="processing-list">' + ''.join(f'<li><strong>{esc(title)}</strong><p>{esc(self.content["processing_p"][i])}</p></li>' for i, title in enumerate(self.content["processing_names"])) + '</ul>' + self.normalization_examples()
        features, offset = [], 0
        for i, group in enumerate(self.features["groups"]):
            entries = []
            for index in range(offset, offset + group["count"]):
                name, label = self.features["names"][index], self.content["feature_labels"][index]
                entries.append(f'<li data-feature-index="{index}"><code>{esc(name)}</code><span class="feature-meaning">{esc(label)}</span></li>')
            features.append(f'<details id="feature-group-{i+1}"><summary>{esc(self.content["feature_group_names"][i])}<span class="feature-count">{group["count"]} {self.t("feature_count")}</span></summary><p class="feature-note">{esc(self.content["feature_group_p"][i])}</p><ul>{"".join(entries)}</ul></details>')
            offset += group["count"]
        feature_body = self.stage(2) + self.p("features_intro", "section-intro") + f'<div class="feature-groups">{"".join(features)}</div>' + f'<p class="caption">{self.link("../../reproducibility/FEATURES.json",self.content["feature_source"],download=True)}</p>'
        architecture = self.stage(3) + self.p("architecture_intro", "section-intro") + self.diagram()
        config = self.stage(5) + self.p("configuration_intro", "section-intro") + self.configuration_table() + self.p("method_scope", "caption")
        return self.section("processing", "processing_title", processing, "01") + self.section("roles", "roles_title", roles, "02", True) + self.section("features", "features_title", feature_body, "03") + self.section("architecture", "architecture_title", architecture, "04", True) + self.section("aggregation", "model_title", self.aggregation(), "05") + self.section("configuration", "configuration_title", config, "06")

    def configuration_table(self) -> str:
        m = self.model
        configs = [m["model_revision"], ", ".join(map(str, m["seeds"])), m["selected_epoch"], f'{m["micro_batch"]} / {m["effective_batch"]}', f'{m["encoder_lr"]} / {m["head_lr"]}', f'{m["scheduler_epochs"]} / {m["warmup_fraction"]}', f'{m["weight_decay"]} / {m["gradient_clip"]}', m["dropout"], self.content["precision_training"], self.content["precision_inference"]]
        rows = ''.join(f'<tr><th scope="row" class="config-key">{esc(label)}</th><td class="config-value">{esc(value)}</td></tr>' for label, value in zip(self.content["config_labels"], configs))
        return '<div class="table-wrap config-table"><table class="table"><tbody>' + rows + '</tbody></table></div>'

    def method_resources(self) -> str:
        """Readable inline resources; all values are rendered locally, with no fetch."""
        m = self.model
        def block(zh, en, zp, ep):
            return f'<article><h3>{self.tr(zh,en)}</h3><p>{self.tr(zp,ep)}</p></article>'
        overview = block("输入", "Input", "一对零售商商品记录。标题、品牌、原生型号和类别组成文本输入；描述、价格、币种及缺失状态通过比较特征参与评分。", "A pair of retailer listings. Title, brand, native model number and category form the text input; description, price, currency and missing-field states contribute through comparison features.")
        overview += block("模型结构", "Architecture", f'RoBERTa-base联合编码商品对，将768维文本表示投影为64维，与{m["feature_count"]}项比较特征连接。分类头为106 → 64 → 1，输出一个原始分数（logit）；训练更新编码器与分类头。', f'RoBERTa-base jointly encodes the pair and projects its 768-dimensional text representation to 64 dimensions. This is concatenated with {m["feature_count"]} comparison features. A 106 → 64 → 1 classifier produces a raw score (logit); training updates both encoder and classifier.')
        overview += block("评分与判断", "Scoring and decision", f'每个种子先对AB、BA两个记录顺序的logit取均值，再用sigmoid转成概率。对种子{self.context["seeds"]}的概率等权平均；最终分数不小于{self.context["threshold"]}时判为匹配。', f'Within each seed, average logits for the AB and BA record orders, then apply sigmoid to obtain a probability. Average probabilities equally across seeds {self.context["seeds"]}. A final score at or above {self.context["threshold"]} is classified as a match.')
        model_body = self.p("model_definition") + f'<div class="resource-summary-grid">{overview}</div>'
        model_body += f'<div class="resource-reading-links">{self.link("#architecture", self.tr("查看结构图", "Architecture diagram"))}{self.link("#features", self.tr("查看42项特征", "The 42 features"))}</div>' + self.p("method_scope", "caption")
        config_body = self.p("configuration_intro") + self.configuration_table()
        config_body += f'<p class="resource-cutoff"><strong>{self.tr("校准阈值", "Calibration cutoff")}</strong><code>{self.context["threshold"]}</code></p>'
        checkpoints = ''.join(f'<li><strong>{self.tr("种子", "Seed")} {item["seed"]}</strong><span>{item["bytes"]:,} bytes</span><code>{esc(item["sha256"])}</code></li>' for item in m["checkpoints"])
        config_body += f'<details class="checkpoint-details"><summary>{self.tr("检查点身份（SHA-256）", "Checkpoint identities (SHA-256)")}</summary><p>{self.tr("哈希用于核对本地权重是否对应这些检查点。公开仓库不分发模型权重。", "Hashes identify the local weights for these checkpoints. Model weights are not redistributed in the public repository.")}</p><ul>{checkpoints}</ul></details>'
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
        return f'<figure class="metric-figure" id="plot-{metric}" data-metric-plot="{metric}" data-axis-min="0" data-axis-max="1" aria-labelledby="plot-title-{metric}"><h3 id="plot-title-{metric}">{self.t(title_key)}</h3><div class="chart-axis" aria-hidden="true"><span>0</span><span>0.25</span><span>0.50</span><span>0.75</span><span>1.00</span></div>{"".join(rows)}<figcaption class="chart-foot">{self.t("chart_caption")}</figcaption></figure>'

    def results_page(self) -> str:
        controls = f'<div class="metric-toolbar"><div class="metric-options" role="group" aria-label="{self.t("metric_control")}"><a class="metric-button" role="button" href="?metric=ap#comparison" data-metric="ap" aria-pressed="false">AP</a><a class="metric-button" role="button" href="?metric=f1#comparison" data-metric="f1" aria-pressed="false">F1</a></div><p>{self.t("chart_range")}</p></div>'
        plots = self.p("comparison_intro", "section-intro") + self.p("reference_comparison_explanation", "section-intro") + controls + self.plot("ap") + self.plot("f1") + f'<noscript><p class="noscript-note">{self.t("nojs")}</p></noscript>'
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
        matrix = f'<div class="matrix-and-note">{matrix}<div class="prose"><h3>{self.t("matrix_note_title")}</h3>{self.p("matrix_note")}<h3>{self.t("delta_title")}</h3>{self.p("delta_p")}</div></div>'
        definitions = '<div class="metric-definition-list">' + ''.join(f'<article><h3>{label}</h3><p>{esc(text)}</p></article>' for label, text in zip(("AP", "F1", "P@100"), self.content["definition_p"])) + '</div>'
        limits = f'<div class="prose">{self.p("limits_p")}{self.link(self.doc("LIMITATIONS"),self.content["pages"]["results"]["links"][1])}</div>'
        return self.section("comparison", "comparison_title", plots, "01") + self.section("gains", "delta_title", self.gains(), "02") + self.section("full-table", "table_title", table, "03") + self.section("confusion", "matrix_title", matrix + self.outcomes(), "04", True) + self.section("metrics", "definitions_title", definitions, "05") + self.section("limitations", "limits_title", limits, "06", True)

    def gains(self) -> str:
        rows = []
        for reference in ("L42", "C0"):
            for key in ("ap", "f1"):
                delta = 100 * (self.results['metrics']['C+'][key] - self.results['metrics'][reference][key])
                label = f'{key.upper()} · C+ − {reference}'
                rows.append(f'<div class="delta-row" data-delta-reference="{reference}" data-delta-metric="{key}" data-value="{delta}"><span>{label}</span><span class="delta-track" aria-hidden="true"><span class="delta-bar" style="width:{delta/6*100:.8f}%"></span></span><strong>+{delta:.4f} {self.tr("百分点","pp")}</strong></div>')
        axes = '<div class="chart-axis delta-axis"><span>0</span><span>2</span><span>4</span><span>6</span></div>'
        intro = self.tr("以下是同一固定样本中指标的绝对差，统一使用0—6个百分点坐标。没有置信区间，条形长度不能用来判断统计显著性。C0为独立训练的零特征网络，不是推断时临时遮挡C+。", "These are absolute metric differences on the same fixed pool, shown on a shared 0–6 percentage-point scale. No confidence intervals are available; bar length does not establish statistical significance. C0 is a separately trained zero-feature network, not C+ masked at inference.")
        return f'<p class="section-intro">{intro}</p><figure class="delta-chart" data-axis-min="0" data-axis-max="6">{axes}{"".join(rows)}<figcaption class="caption">{self.t("result_interpretation")}</figcaption></figure>'

    def outcomes(self) -> str:
        cm = self.results['metrics']['C+']['confusion']
        rows = []
        for group, correct, error, zh, en, ez, ee in [('positive','tp','fn','实际同一商品','Actual matches','漏匹配','Missed matches'),('negative','tn','fp','实际不同商品','Actual non-matches','误合并','False matches')]:
            total = cm[correct] + cm[error]
            rows.append(f'<div class="outcome-row" data-outcome="{group}"><h3>{self.tr(zh,en)} · {total}</h3><div class="outcome-bar" aria-hidden="true"><span class="outcome-correct" style="width:{100*cm[correct]/total:.8f}%"></span><span class="outcome-error" style="width:{100*cm[error]/total:.8f}%"></span></div><div class="outcome-labels"><span>{self.tr("正确","Correct")}: {cm[correct]} ({100*cm[correct]/total:.2f}%)</span><strong>{self.tr(ez,ee)}: {cm[error]} ({100*cm[error]/total:.2f}%)</strong></div></div>')
        reference = self.results['metrics']['L42']['confusion']
        tradeoff = self.tr(f'相较L42，误合并总数由{reference["fp"]}降到{cm["fp"]}，漏匹配由{reference["fn"]}变为{cm["fn"]}。这些是聚合计数变化，不代表已经定位出具体哪些商品对被修正；F1提升伴随少量召回下降。', f'Compared with L42, false matches fell from {reference["fp"]} to {cm["fp"]}; missed matches changed from {reference["fn"]} to {cm["fn"]}. These aggregate counts do not identify which particular pairs were corrected. Higher F1 comes with a small recall decrease.')
        label = self.tr("各真实类别中的正确与错误比例","Correct and incorrect outcomes within each actual class")
        return f'<div class="outcome-chart"><h3>{label}</h3><p>{self.tr("每条归一化为100%，两条的分母不同：190与365。","Each bar is normalized to 100%; the denominators differ: 190 and 365.")}</p>{"".join(rows)}<p class="caption">{tradeoff}</p></div>'

    def command_flow(self) -> str:
        stages = [
            ('project-role', '保留ID、业务视图和原生字段', 'Kept IDs, business views and native fields', '按固定用途投影，拒绝多余字段', 'Project a fixed role; reject extra fields', '用途明确的商品对JSON', 'Role-specific pair JSON'),
            ('prepare', '商品对JSON；非fit复用训练IDF', 'Pair JSON; reuse training IDF outside fit', '构造x42与AB/BA token；非fit输入不带标签', 'Construct x42 and AB/BA tokens; no target labels in non-fit inputs', '特征、token与准备回执', 'Features, tokens and preparation receipt'),
            ('train', '训练用途输入、可信模型快照、种子', 'Fit input, trusted snapshot and seed', '更新编码器和分类头，保存各轮检查点', 'Update encoder and classifier; save epoch checkpoints', '检查点及训练记录', 'Checkpoints and training records'),
            ('predict / select-epoch', '检查点、开发输入及独立开发标签', 'Checkpoints, dev input and separate dev labels', '先预测，再依pair ID接标签选择共同轮数', 'Predict first, then join dev labels by pair ID to select a shared epoch', '预测分数与共同E4', 'Prediction scores and shared E4'),
            ('calibrate / evaluate', '校准分数与校准标签；评价分数与评价标签', 'Cal scores/labels; evaluation scores/labels', '校准锁定阈值，评价复用该阈值', 'Freeze the cutoff on calibration; reuse it for evaluation', '阈值文件与聚合指标', 'Threshold artifact and aggregate metrics'),
        ]
        body = ''.join(f'<tr><th scope="row"><code>{name}</code></th><td>{self.tr(iz,ie)}</td><td>{self.tr(oz,oe)}</td><td>{self.tr(rz,re)}</td></tr>' for name,iz,ie,oz,oe,rz,re in stages)
        heads = ''.join(f'<th scope="col">{self.tr(z,e)}</th>' for z,e in [('命令阶段','Command stage'),('输入','Input'),('处理','Operation'),('输出','Output')])
        return f'<p class="section-intro">{self.tr("以下命令连接商品对准备、模型训练、预测与评价。实际运行需要本地商品输入、固定依赖及模型文件；命令指南给出了各阶段的文件格式和完整参数。","These commands connect pair preparation, model training, prediction and evaluation. Model runs require local product inputs, pinned dependencies and model files; the CLI guide specifies the file formats and complete arguments for each stage.")}</p><div class="table-wrap"><table><thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table></div>{self.link(self.doc("CORE_USAGE"),self.content["pages"]["reproduce"]["links"][1])}'

    def toc(self, page: str) -> str:
        keys = {
            'index':[('project','brief_title'),('evidence','overview_choice_title'),('example','overview_example_title'),('summary','overview_result_title'),('public-code','overview_close_title')],
            'method':[('processing','processing_title'),('roles','roles_title'),('features','features_title'),('architecture','architecture_title'),('aggregation','model_title'),('configuration','configuration_title')],
            'results':[('comparison','comparison_title'),('gains','delta_title'),('full-table','table_title'),('confusion','matrix_title'),('metrics','definitions_title'),('limitations','limits_title')],
            'reproduce':[('quickstart','quickstart_title'),('public-checks','public_title'),('command-flow','command_flow_title'),('replay','replay_title'),('fresh-training','fresh_title'),('resources','resources_title'),('rebuild','rebuild_title')],
        }[page]
        links = ''.join(f'<a href="#{anchor}"><span>{i+1:02}</span>{self.t(key) if key!="command_flow_title" else self.tr("命令的输入与输出","Command inputs and outputs")}</a>' for i,(anchor,key) in enumerate(keys))
        return f'<p class="toc-label">{self.tr("本页内容","On this page")}</p><nav aria-label="{self.tr("页内目录","Page contents")}">{links}</nav>'

    def command(self, identity: str, lines: list[str]) -> str:
        text = "\n".join(lines)
        return f'<div class="command-block"><pre id="{identity}"><code>{esc(text)}</code></pre><button class="copy-button" type="button" data-copy-target="{identity}" data-copy-status="{identity}-status" hidden>{self.t("copy")}</button></div><span class="copy-status" id="{identity}-status" role="status" aria-live="polite"></span>'

    def reproduce(self) -> str:
        setup = self.p("quickstart_intro", "section-intro")
        setup += self.command("clone-command", ["git clone https://github.com/Yemyu/product-entity-matching.git", "cd product-entity-matching"])
        setup += '<h3>macOS / Linux</h3>' + self.command("setup-unix", ["python3 -m venv .venv", "source .venv/bin/activate", "python scripts/public_smoke.py", "python -m http.server 8000 --bind 127.0.0.1"])
        setup += '<h3>Windows · PowerShell</h3>' + self.command("setup-windows", ["py -3 -m venv .venv", r".\.venv\Scripts\python.exe scripts/public_smoke.py", r".\.venv\Scripts\python.exe -m http.server 8000 --bind 127.0.0.1"])
        setup += self.p("quickstart_preview") + self.link("http://127.0.0.1:8000/showcase/"+self.language+"/index.html", self.content["quickstart_open"])
        setup += self.p("quickstart_python", "caption")
        setup += self.p("quickstart_gpu") + self.link(self.doc("REPRODUCIBILITY"),self.content["quickstart_gpu_link"])
        steps = []
        for i, (title, text, commands) in enumerate([
            ("check_step_title", "check_step_p", ["python scripts/public_smoke.py", "python scripts/run_notebook.py", "python scripts/verify_public_release.py"]),
            ("tests_step_title", "tests_step_p", ["python scripts/matching.py --help", "python scripts/run_tests.py"]),
        ], 1):
            steps.append(f'<article class="reproduction-step"><div class="step-index" aria-hidden="true">0{i}</div><div><h3>{self.t(title)}</h3>{self.p(text)}{self.command("public-command-"+str(i),commands)}</div></article>')
        public = self.p("public_intro", "section-intro") + "".join(steps)
        replay = self.p("replay_intro", "section-intro") + '<ul class="replay-list">' + ''.join(f'<li>{esc(item)}</li>' for item in self.content["replay_items"]) + '</ul>' + self.p("replay_note", "caption")
        fresh = f'<div class="prose">{self.p("fresh_p")}{self.p("fresh_note")}</div>'
        resource_urls = [self.doc(name) for name in DOCS] + ["../../results/fixed-results.json", "../../reproducibility/FINAL_MODEL.json", "../../reproducibility/FEATURES.json"]
        resources = '<div class="resource-list">' + ''.join(f'<a href="{url}"><span>{esc(label)}</span><span aria-hidden="true">↗</span></a>' for url, label in zip(resource_urls, self.content["resource_names"])) + '</div>'
        resources += f'<p class="caption">{self.link("https://github.com/Yemyu/product-entity-matching/tree/main/src/product_matching",self.content["source_project"])}</p>'
        rebuild = self.p("rebuild_p", "section-intro") + self.command("build-command", ["python scripts/build_showcase.py", "python scripts/build_showcase.py --check"]) + self.p("rebuild_check_p", "caption")
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
            "results": ("../assets/fixed-results.csv", self.doc("LIMITATIONS")),
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
