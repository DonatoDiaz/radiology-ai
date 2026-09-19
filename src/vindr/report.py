"""Radiology knowledge base and protocol-style text report.

Glossary entries are distilled from standard radiology teaching notes
(lung CT / X-ray semiotics) and are deliberately phrased as radiological
descriptions, not diagnoses. The auto-generated protocol extends the
capabilities of the healthcare practitioner on site, so that they can
interpret a study at the level of a specialist without escalation.
"""

from __future__ import annotations

import datetime as _dt

from vindr.i18n import label_name

# Radiological description per finding, keyed by original English label.
GLOSSARY: dict[str, dict[str, str]] = {
    "No finding": {
        "ru": "Патологических изменений в лёгочной ткани не выявлено",
        "en": "No pathological changes detected in the pulmonary tissue",
        "zh": "未发现肺组织病理改变",
    },
    "Atelectasis": {
        "ru": "Спадение лёгочной ткани, повышение её плотности; при сохранении секрета в бронхах определяется симптом воздушной бронхографии (компрессионный ателектаз), при его отсутствии — вероятна обтурация бронха",
        "en": "Collapse of lung tissue with increased density; an air bronchogram sign indicates compression atelectasis, its absence suggests bronchial obstruction",
        "zh": "肺组织萎陷、密度增高；可见空气支气管征提示压迫性肺不张，未见则提示支气管阻塞",
    },
    "Consolidation": {
        "ru": "Повышение плотности с полной облитерацией воздушных альвеолярных пространств; сосуды на этом фоне не прослеживаются; типичная треугольная тень с основанием к плевре, характерная для долевой консолидации (пневмонии)",
        "en": "Increased density with complete obliteration of alveolar air spaces; vessels are no longer visible; typical triangular shadow with its base toward the pleura, consistent with lobar consolidation (pneumonia)",
        "zh": "密度增高并肺泡含气腔隙完全闭塞，其内血管影消失；典型基底朝向胸膜的三角形致密影，符合大叶性实变（肺炎）",
    },
    "Infiltration": {
        "ru": "Инфильтративные изменения лёгочной ткани: участки уплотнения с нечёткими контурами, окружённые зоной перифокального воспаления; требует дифференциальной диагностики (пневмония, туберкулёз и др.)",
        "en": "Infiltrative lung changes: areas of density with ill-defined borders surrounded by perifocal inflammation; differential diagnosis required (pneumonia, tuberculosis, etc.)",
        "zh": "肺组织浸润性改变：边界不清的致密影，周围伴炎性反应；需鉴别诊断（肺炎、肺结核等）",
    },
    "Lung Opacity": {
        "ru": "Повышение плотности по типу «матового стекла»: сохранение визуализации сосудов и структур долек на фоне пеноподобного уплотнения (заполнение альвеол жидкостью, утолщение межальвеолярного интерстиция)",
        "en": "Ground-glass opacity: vessels and lobular structures remain visible against a hazy density (alveolar filling and thickening of the interalveolar interstitium)",
        "zh": "磨玻璃样致密影：可见血管及小叶结构（肺泡内渗出、小叶间隔增厚）",
    },
    "Nodule/Mass": {
        "ru": "Очаговое образование: очаг — уплотнение до 1 см; узел/масса — более крупное образование; при бугристых нечётких контурах и росте показано дообследование (КТ, при размере более ~1 см — морфологическая верификация)",
        "en": "Focal lesion: a focus is a density up to 1 cm; a nodule/mass is larger; lobulated or irregular borders and growth warrant further work-up (CT, biopsy when larger than ~1 cm)",
        "zh": "局灶性病变：病灶为1厘米以内致密影；结节/肿块较大；边缘分叶、不规则或生长时需进一步检查（CT、大于约1厘米时行病理活检）",
    },
    "Other lesion": {
        "ru": "Прочие изменения лёгочной ткани, не отнесённые к перечисленным категориям; требуется уточнение по клинической картине",
        "en": "Other lung changes not assigned to the listed categories; clarify against the clinical presentation",
        "zh": "其他未归类的肺组织改变，需结合临床表现进一步明确",
    },
    "Pleural effusion": {
        "ru": "Скопление жидкости в плевральной полости (притупление синуса, затенение в отлогих отделах); при присоединении инфекции — эмпиема плевры",
        "en": "Fluid accumulation in the pleural space (blunted costophrenic angle, dependent opacity); empyema when secondarily infected",
        "zh": "胸膜腔内积液（肋膈角变钝、较低位致密影）；继发感染时为脓胸",
    },
    "Pleural thickening": {
        "ru": "Утолщение париетальной плевры, отслойка/шварты, субплевральные линейные плотности параллельно грудной стенке",
        "en": "Thickening of the parietal pleura, pleural plaques, subpleural linear densities running parallel to the chest wall",
        "zh": "壁层胸膜增厚、胸膜斑片、与胸壁平行的胸膜下线样致密影",
    },
    "Pneumothorax": {
        "ru": "Воздух в плевральной полости с коллапсом лёгкого, зона резкого снижения плотности без лёгочного рисунка",
        "en": "Air in the pleural space with lung collapse; a hyperlucent zone without pulmonary markings",
        "zh": "胸膜腔积气伴肺萎陷；无肺纹理的透亮区",
    },
    "Pulmonary fibrosis": {
        "ru": "Фиброзные изменения интерстиция: утолщение внутридолькового интерстиция, линейные и сетчатые плотности, тракционные бронхоэктазы, в исходе — кистозный паттерн («пчелиные соты»)",
        "en": "Interstitial fibrosis: thickening of the intralobular interstitium, linear and reticular densities, traction bronchiectasis, eventually a honeycombing pattern",
        "zh": "间质纤维化：小叶间隔增厚、线样及网状致密影、牵拉性支气管扩张，终末期蜂窝肺改变",
    },
}

# Header rows of the protocol report, per language.
PROTOCOL_HEADERS: dict[str, dict[str, str]] = {
    "title": {
        "ru": "ПРОТОКОЛ АВТОМАТИЧЕСКОЙ ОЦЕНКИ РЕНТГЕНОГРАММЫ ОРГАНОВ ГРУДНОЙ КЛЕТКИ",
        "en": "AUTOMATED CHEST X-RAY ASSESSMENT PROTOCOL",
        "zh": "胸部X线自动评估报告",
    },
    "description": {
        "ru": "ОПИСАНИЕ",
        "en": "DESCRIPTION",
        "zh": "描述",
    },
    "conclusion": {
        "ru": "ЗАКЛЮЧЕНИЕ",
        "en": "CONCLUSION",
        "zh": "结论",
    },
    "recommendations": {
        "ru": "РЕКОМЕНДАЦИИ",
        "en": "RECOMMENDATIONS",
        "zh": "建议",
    },
    "disclaimer": {
        "ru": "Результат сформирован ИИ-ассистентом, который помогает специалисту интерпретировать исследование на уровне профильного рентгенолога. Используйте его вместе с клинической картиной.",
        "en": "This result was generated by an AI assistant that helps the practitioner interpret the study at the level of a specialist radiologist. Combine it with the clinical picture.",
        "zh": "本结果由AI助手生成，辅助专业人员以专科影像医师的水平解读影像。请结合临床情况使用。",
    },
}

# Differential-diagnosis rules: (label1, label2, min_prob, text).
DIFFERENTIAL_RULES: list[tuple[str, str, float, dict[str, str]]] = [
    (
        "Consolidation",
        "Lung Opacity",
        0.5,
        {
            "ru": "Совместно плотность «матового стекла» и консолидация — альвеолярный паттерн, характерный для пневмонии",
            "en": "Ground-glass opacity together with consolidation suggests an alveolar pattern, typical of pneumonia",
            "zh": "磨玻璃样致密影伴实变为肺泡性改变，符合肺炎表现",
        },
    ),
    (
        "Consolidation",
        "Pleural effusion",
        0.4,
        {
            "ru": "Консолидация с плевральным выпотом — пневмония, осложнённая плевритом/эмпиемой",
            "en": "Consolidation with pleural effusion suggests pneumonia complicated by pleurisy/empyema",
            "zh": "实变伴胸腔积液提示肺炎合并胸膜炎/脓胸",
        },
    ),
    (
        "Nodule/Mass",
        "",
        0.7,
        {
            "ru": "Высокая вероятность узла/массы — при размере > ~1 см и росте показана морфологическая верификация на базе учреждения",
            "en": "High probability of a nodule/mass — biopsy/verification on-site when larger than ~1 cm or growing",
            "zh": "结节/肿块概率高 — 大于约1厘米或增大时应在院内行病理活检",
        },
    ),
    (
        "Infiltration",
        "Consolidation",
        0.4,
        {
            "ru": "Инфильтрация с консолидацией — гнойно-инфильтративный процесс; дифференцировать пневмонию и инфильтративный туберкулёз",
            "en": "Infiltration with consolidation — infiltrative process; differentiate pneumonia from infiltrative tuberculosis",
            "zh": "浸润伴实变 — 浸润性病变；需鉴别肺炎与浸润型肺结核",
        },
    ),
    (
        "Pulmonary fibrosis",
        "Lung Opacity",
        0.4,
        {
            "ru": "Фиброзные изменения с «матовым стеклом» — фиброзирующее интерстициальное заболевание",
            "en": "Fibrosis with ground-glass opacity — fibrosing interstitial disease",
            "zh": "纤维化伴磨玻璃影 — 纤维化性间质性肺病",
        },
    ),
    (
        "Pleural effusion",
        "Atelectasis",
        0.4,
        {
            "ru": "Выпот с ателектазом — вероятен компрессионный ателектаз при сохраняющейся проходимости бронха",
            "en": "Effusion with atelectasis — likely compression atelectasis with patent bronchi",
            "zh": "积液伴肺不张 — 可能为压迫性肺不张（支气管通畅）",
        },
    ),
    (
        "Pneumothorax",
        "",
        0.6,
        {
            "ru": "Высокая вероятность пневмоторакса — при клинике напряжённого пневмоторакса необходимо экстренное пособие на месте",
            "en": "High probability of pneumothorax — if a tension pneumothorax is suspected, emergency care on site",
            "zh": "气胸概率高 — 若怀疑张力性气胸需当场紧急处理",
        },
    ),
]


def glossary_entry(label: str, lang: str) -> str:
    return GLOSSARY.get(label, {}).get(lang, "")


def differential_hints(
    probs: list[float] | tuple[float, ...], labels: list[str], lang: str
) -> list[str]:
    """Return matching differential-diagnosis hints for the prediction."""
    idx = {lbl: i for i, lbl in enumerate(labels)}
    hints: list[str] = []
    for lbl1, lbl2, thr, text in DIFFERENTIAL_RULES:
        p1 = probs[idx[lbl1]] if lbl1 in idx else 0.0
        p2 = probs[idx[lbl2]] if lbl2 and lbl2 in idx else 1.0
        if p1 >= thr and p2 >= thr:
            hints.append(text.get(lang, text["en"]))
    return hints


def render_protocol(
    probs: list[float] | tuple[float, ...],
    labels: list[str],
    lang: str,
    image_path: str = "",
    top_k: int = 5,
    threshold: float = 0.1,
) -> str:
    """Render findings + differential hints as a protocol-style report."""
    h = {k: v[lang] for k, v in PROTOCOL_HEADERS.items()}
    no_finding = "No finding"
    order = sorted(range(len(labels)), key=lambda i: probs[i], reverse=True)

    findings = [
        i for i in order if labels[i] != no_finding and probs[i] > threshold
    ] or [i for i in order if labels[i] == no_finding]

    if len(findings) == 1 and labels[findings[0]] == no_finding:
        summary = (
            "патологии не выявлено / no pathology / 未发现异常"
            if lang == "en"
            else "暂无" if lang == "zh" else "патологии не выявлено"
        )
    else:
        summary = ", ".join(
            f"{label_name(labels[i], lang)} ({probs[i]:.2f})" for i in findings
        )

    date = _dt.date.today().isoformat()
    lines = [
        h["title"],
        f"  Lung Radiology AI (EfficientNet-B0) | date: {date}",
        f"  image: {image_path}",
        "",
        h["description"],
    ]
    for i in findings:
        prob, lbl = probs[i], labels[i]
        gloss = glossary_entry(lbl, lang)
        lines.append(
            f"  - {label_name(lbl, lang)}: {prob:.3f}"
            + (f" — {gloss}" if gloss else "")
        )
    lines += [
        "",
        h["conclusion"],
        f"  {summary}",
    ]
    hints = differential_hints(probs, labels, lang)
    if hints:
        lines += [
            "",
            h["recommendations"],
        ]
        lines += [f"  - {t}" for t in hints]
        lines.append(
            {
                "ru": "  - Результат использовать вместе с клиническими данными.",
                "en": "  - Use the result together with the clinical data.",
                "zh": "  - 请结合临床数据使用本结果。",
            }[lang]
        )
    lines += [
        "",
        h["disclaimer"],
    ]
    return "\n".join(lines)