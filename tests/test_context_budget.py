from src.ai.context_budget import ContextBudget


def make_messages(turns):
    messages = [{"role": "system", "content": "System prompt"}]

    for index, content in enumerate(turns):
        messages.extend([
            {"role": "user", "content": f"Question {index}"},
            {"role": "assistant", "content": content},
        ])

    return messages


def test_trim_history_preserves_system_and_latest_question():
    messages = make_messages(["A" * 1000, "B" * 1000, "Latest question"])

    trimmed = ContextBudget.trim_history(messages, max_messages=4)

    assert trimmed[0] == messages[0]
    assert trimmed[-2]["content"] == "Question 2"
    assert trimmed[-1]["content"] == "Latest question"


def test_trim_history_preserves_complete_tool_turns():
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "Old question"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_old"}],
        },
        {
            "role": "tool",
            "tool_call_id": "call_old",
            "content": "Old result",
        },
        {"role": "assistant", "content": "Old answer"},
        {"role": "user", "content": "Latest question"},
        {"role": "assistant", "content": "Latest answer"},
    ]

    trimmed = ContextBudget.trim_history(messages, max_messages=4)

    assert trimmed[0] == messages[0]
    assert all(
        item.get("content") != "Old result"
        for item in trimmed
    )
    assert trimmed[-2]["content"] == "Latest question"


def test_token_aware_trim_preserves_latest_question():
    messages = make_messages([
        "A" * 3000,
        "B" * 3000,
        "Latest question",
    ])

    trimmed = ContextBudget.trim_to_budget(
        messages,
        max_tokens=1200,
    )

    assert trimmed[0] == messages[0]
    assert trimmed[-1]["content"] == "Latest question"
    assert ContextBudget.estimate_tokens(trimmed) <= 1200


def test_token_aware_trim_rejects_oversized_current_turn():
    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "X" * 10000},
    ]

    try:
        ContextBudget.trim_to_budget(messages, max_tokens=1200)
    except ValueError as exc:
        assert "current turn" in str(exc).lower()
    else:
        raise AssertionError("Expected oversized current turn to be rejected")
