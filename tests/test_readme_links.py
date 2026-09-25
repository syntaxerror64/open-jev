# tests/test_readme_links.py
from __future__ import annotations
import re, subprocess
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def _local_targets() -> list[str]:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    return [
        m.group(1).split("#")[0]
        for m in LINK.finditer(text)
        if not m.group(1).startswith(("http://", "https://", "mailto:", "#"))
        and m.group(1).split("#")[0]
    ]


def test_readme_local_links_resolve() -> None:
    # После снятия ссылки на docs список пуст и проверка проходит вакуумно —
    # это нормально: тест оживёт, как только в README появятся локальные ссылки.
    for rel in _local_targets():
        assert (REPO / rel).exists(), f"мёртвая ссылка в README: {rel}"


def test_readme_local_links_are_published() -> None:
    """Локальная ссылка обязана жить на GitHub, а не только в рабочей копии."""
    for rel in _local_targets():
        proc = subprocess.run(
            ["git", "check-ignore", "-q", str(REPO / rel)],
            capture_output=True,
        )
        if proc.returncode not in (0, 1):
            pytest.skip("git недоступен — проверка игнорирования пропущена")
        assert proc.returncode == 1, (
            f"README ссылается на {rel}, но тот в .gitignore — на GitHub будет 404"
        )


def test_readme_demo_command_is_real() -> None:
    text = (REPO / "README.md").read_text(encoding="utf-8")
    assert "python open_jev/main.py" not in text, "в main.py нет блока __main__"
    assert "python jev.py" not in text, "jev.py не существует"
    assert "python example.py" in text, "должна быть рабочая команда демо"
