from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from _testlib import Check, run_all  # noqa: E402


POLICY = ROOT / "docs/CLINICIAN_LANGUAGE_POLICY.md"
CLINICIAN_DOCUMENTS = (
    ROOT / "docs/CURRENT_MODEL.md",
    ROOT / "docs/results/universal_clinical_router_v4.md",
    ROOT / "docs/results/universal_clinical_router_v5_candidate.md",
)
TECHNICAL_NOTE = "## 技术备注"


def _clinician_body(path: Path) -> str:
    text = path.read_text()
    if TECHNICAL_NOTE not in text:
        raise ValueError(f"{path.name} 缺少技术备注分界")
    return text.split(TECHNICAL_NOTE, 1)[0]


def test_chinese_policy_is_explicit_and_machine_identifiers_are_separate(c: Check):
    c.true(POLICY.is_file(), "缺少医生材料中文规范")
    text = POLICY.read_text()
    c.true("医生可见正文必须使用中文" in text)
    c.true("代码标识" in text and "技术备注" in text)
    c.true("不得出现未解释的英文缩写" in text)
    c.true("docs/CLINICIAN_LANGUAGE_POLICY.md" in (ROOT / "docs/PIPELINE.md").read_text())


def test_current_clinician_documents_have_no_english_prose(c: Check):
    for path in CLINICIAN_DOCUMENTS:
        text = path.read_text()
        c.true(text.startswith("<!-- 面向中国医生 -->\n"), path.name)
        c.true("## 名词说明" in text, path.name)
        body = _clinician_body(path)
        match = re.search(r"[A-Za-z]{2,}", body)
        c.eq(match, None, f"{path.name} 医生正文残留英文：{match.group(0) if match else ''}")
        c.true(len(re.findall(r"[\u4e00-\u9fff]", body)) >= 200, path.name)


if __name__ == "__main__":
    run_all("test_clinician_facing_chinese_v1", dict(globals()))
