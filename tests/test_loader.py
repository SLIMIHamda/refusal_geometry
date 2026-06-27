import pytest

from asw.models.loader import quant_spec


def test_quant_spec_none():
    assert quant_spec(None) is None
    assert quant_spec("bf16") is None        # bf16 is "no quantization"


def test_quant_spec_int8():
    assert quant_spec("int8") == {"load_in_8bit": True}


def test_quant_spec_nf4():
    s = quant_spec("nf4")
    assert s["load_in_4bit"] is True
    assert s["bnb_4bit_quant_type"] == "nf4"
    assert s["bnb_4bit_use_double_quant"] is True


def test_quant_spec_unknown_raises():
    with pytest.raises(ValueError):
        quant_spec("fp7")
