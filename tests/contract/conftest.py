import pytest

class FakeLLM:
    """Deterministic stand-in for OpenRouterClient. Returns queued responses."""
    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []
    def chat(self, prompt, system=None):
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("FakeLLM ran out of queued responses")
        return self._responses.pop(0)

@pytest.fixture
def fake_llm():
    return FakeLLM
