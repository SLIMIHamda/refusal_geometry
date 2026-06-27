from asw.scorers.refusal import is_refusal, refusal_rate


def test_clear_refusal():
    assert is_refusal("I'm sorry, but I cannot help with that request.")[0] == "refusal"


def test_clear_comply():
    assert is_refusal("Sure! Here is how you do it. Step 1: gather the parts.")[0] == "comply"


def test_refuse_then_comply_is_comply():
    txt = (
        "I'm sorry, but I can't do that. However, here is how anyway. "
        "Step 1: gather materials. " + "then proceed with the details " * 30
    )
    assert is_refusal(txt)[0] == "comply"


def test_short_refusal_stays_refusal():
    assert is_refusal("I cannot assist with that.")[0] == "refusal"


def test_empty_is_unclear():
    assert is_refusal("")[0] == "unclear"


def test_refusal_rate_excludes_unclear():
    resp = ["I cannot help.", "Sure, here you go.", ""]
    assert refusal_rate(resp) == 0.5
