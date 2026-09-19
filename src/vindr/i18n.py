"""Multilingual label translation for predictions (ru / en / zh)."""

from __future__ import annotations

from vindr.labels import ALL_LABELS, NO_FINDING

# Original English label -> translations.
TRANSLATIONS: dict[str, dict[str, str]] = {
    NO_FINDING: {"en": "No finding", "ru": "Находок нет", "zh": "无异常"},
    "Atelectasis": {"en": "Atelectasis", "ru": "Ателектаз", "zh": "肺不张"},
    "Consolidation": {"en": "Consolidation", "ru": "Консолидация", "zh": "实变"},
    "Emphysema": {"en": "Emphysema", "ru": "Эмфизема", "zh": "肺气肿"},
    "Infiltration": {"en": "Infiltration", "ru": "Инфильтрация", "zh": "浸润"},
    "Lung Opacity": {"en": "Lung Opacity", "ru": "Помутнение лёгочного поля", "zh": "肺野致密影"},
    "Nodule/Mass": {"en": "Nodule/Mass", "ru": "Узел/Масса", "zh": "结节/肿块"},
    "Other lesion": {"en": "Other lesion", "ru": "Другое поражение", "zh": "其他病变"},
    "Pleural effusion": {"en": "Pleural effusion", "ru": "Плевральный выпот", "zh": "胸腔积液"},
    "Pleural thickening": {"en": "Pleural thickening", "ru": "Утолщение плевры", "zh": "胸膜增厚"},
    "Pneumothorax": {"en": "Pneumothorax", "ru": "Пневмоторакс", "zh": "气胸"},
    "Pulmonary fibrosis": {"en": "Pulmonary fibrosis", "ru": "Лёгочный фиброз", "zh": "肺纤维化"},
    "Ill-defined opacity": {"en": "Ill-defined opacity", "ru": "Неоднородное помутнение", "zh": "边界不清致密影"},
    "Parenchymal bands": {"en": "Parenchymal bands", "ru": "Паренхиматозные тяжи", "zh": "肺实质条索"},
    "Pulmonary cyst": {"en": "Pulmonary cyst", "ru": "Киста лёгкого", "zh": "肺囊肿"},
    "Scarring": {"en": "Scarring", "ru": "Рубцовые изменения", "zh": "瘢痕"},
    "Subcutaneous emphysema": {"en": "Subcutaneous emphysema", "ru": "Подкожная эмфизема", "zh": "皮下气肿"},
    "Lung cavity": {"en": "Lung cavity", "ru": "Полость в лёгком", "zh": "肺空洞"},
    "Aortic enlargement": {"en": "Aortic enlargement", "ru": "Расширение аорты", "zh": "主动脉增宽"},
    "Calcification": {"en": "Calcification", "ru": "Кальциноз", "zh": "钙化"},
    "Cardiomegaly": {"en": "Cardiomegaly", "ru": "Кардиомегалия", "zh": "心脏增大"},
    "ILD": {"en": "ILD", "ru": "Интерстициальное поражение лёгких", "zh": "间质性肺病"},
    "Mediastinal shift": {"en": "Mediastinal shift", "ru": "Смещение средостения", "zh": "纵隔移位"},
    "Tracheal deviation": {"en": "Tracheal deviation", "ru": "Отклонение трахеи", "zh": "气管偏移"},
    "Hilar lymphadenopathy": {"en": "Hilar lymphadenopathy", "ru": "Лимфаденопатия корней лёгких", "zh": "肺门淋巴结肿大"},
    "Clavicle fracture": {"en": "Clavicle fracture", "ru": "Перелом ключицы", "zh": "锁骨骨折"},
    "Rib fracture": {"en": "Rib fracture", "ru": "Перелом ребра", "zh": "肋骨骨折"},
    "Other fracture": {"en": "Other fracture", "ru": "Другой перелом", "zh": "其他骨折"},
}

LANGUAGES = ("en", "ru", "zh")
DEFAULT_LANG = "en"


def label_name(label: str, lang: str = DEFAULT_LANG) -> str:
    """Return `label` translated to `lang`, falling back to the original."""
    return TRANSLATIONS.get(label, {}).get(lang, label)