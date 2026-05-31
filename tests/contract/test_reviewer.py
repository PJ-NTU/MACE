from mace.contract.reviewer import review


def test_review_approved(fake_llm):
    ok, fb = review(fake_llm(["APPROVED"]), "input schema", "desc", "code")
    assert ok and fb is None


def test_review_rejected(fake_llm):
    ok, fb = review(fake_llm(["REJECTED: field 'x' is missing"]),
                    "input schema", "desc", "code")
    assert not ok and "missing" in fb
