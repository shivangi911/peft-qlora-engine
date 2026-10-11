from peft_engine.dataset import (
    DEFAULT_SYSTEM,
    build_chatml,
    extract_prompt,
    format_to_chatml,
    split_rows,
)

CONV = [
    {"from": "system", "value": "Be brief."},
    {"from": "human", "value": "Hi"},
    {"from": "gpt", "value": "Hello"},
]


def test_format_to_chatml():
    assert format_to_chatml(CONV) == (
        "<|im_start|>system\nBe brief.<|im_end|>\n"
        "<|im_start|>user\nHi<|im_end|>\n"
        "<|im_start|>assistant\nHello<|im_end|>\n"
    )


def test_unknown_roles_are_skipped():
    assert format_to_chatml([{"from": "tool", "value": "x"}]) == ""


def test_extract_prompt_uses_system_and_first_human():
    assert extract_prompt(CONV) == {"system": "Be brief.", "user": "Hi"}


def test_extract_prompt_default_system_and_none():
    assert extract_prompt([{"from": "human", "value": "Q"}])["system"] == DEFAULT_SYSTEM
    assert extract_prompt([{"from": "gpt", "value": "A"}]) is None


def test_build_chatml_matches_format():
    assert build_chatml("Be brief.", "Hi", "Hello") == format_to_chatml(CONV)


def test_split_rows():
    train, valid = split_rows(list(range(10)))
    assert len(train) == 9 and len(valid) == 1
