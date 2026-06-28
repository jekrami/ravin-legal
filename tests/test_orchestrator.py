import pytest

from legal_analyzer.orchestrator import run_deep_analysis


def test_run_deep_analysis_filters_passes(monkeypatch):
    monkeypatch.setattr(
        "legal_analyzer.orchestrator.ollama_chat",
        lambda **kwargs: "خروجی تست",
    )

    events = list(
        run_deep_analysis("متن قرارداد", selected_pass_ids=[1, 3])
    )
    progress = [e for e in events if e["event"] == "progress"]
    complete = [e for e in events if e["event"] == "complete"][0]

    assert len(progress) == 2
    assert set(complete["results"].keys()) == {1, 3}


def test_run_deep_analysis_rejects_empty_text():
    with pytest.raises(ValueError, match="empty"):
        list(run_deep_analysis("   "))


def test_run_deep_analysis_rejects_no_passes():
    with pytest.raises(ValueError, match="No analysis passes"):
        list(run_deep_analysis("متن", selected_pass_ids=[]))
