from document_processor import normalize_text, process_document


def test_normalize_text_converts_persian_digits():
    assert normalize_text("مبلغ ۱۲۳") == "مبلغ 123"


def test_normalize_text_standardizes_arabic_characters():
    assert "ی" in normalize_text("علي")
    assert "ک" in normalize_text("كتاب")


def test_process_document_chunks_text(monkeypatch):
    monkeypatch.setattr(
        "document_processor._extract_and_normalize",
        lambda _path: "بند اول قرارداد. " * 50,
    )
    chunks = process_document("fake.pdf")
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
