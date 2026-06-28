import pytest

from llm_handler import LlmHandler


@pytest.mark.asyncio
async def test_both_models_fail_returns_error(monkeypatch):
  handler = LlmHandler()

  def fail(_model, _prompt):
      from legal_analyzer.ollama_client import OllamaError
      raise OllamaError("down")

  monkeypatch.setattr("llm_handler.ollama_generate", fail)
  result = await handler.get_synthesized_answer("سوال", "متن")
  assert "خطا در ارتباط" in result


@pytest.mark.asyncio
async def test_one_model_fail_skips_synthesis(monkeypatch):
  handler = LlmHandler()
  calls = []

  def generate(model, prompt):
      calls.append(model)
      if model.endswith("gemma3"):
          from legal_analyzer.ollama_client import OllamaError
          raise OllamaError("down")
      return "پاسخ تحلیلگر"

  monkeypatch.setattr("llm_handler.ollama_generate", generate)
  result = await handler.get_synthesized_answer("سوال", "متن")
  assert result == "پاسخ تحلیلگر"
  assert all("llama3" not in c for c in calls)
