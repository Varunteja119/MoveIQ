from __future__ import annotations

import json
import math
import time
import uuid

import redis


REQUEST_BUDGET = 7000
COMPLETION_RESERVE = 800
ROLLING_TPM_LIMIT = 7000
ROLLING_WINDOW_SECONDS = 60
MAX_HISTORY_MESSAGES = 24

RESERVE_SCRIPT = """
local key = KEYS[1]

local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local requested = tonumber(ARGV[3])
local limit = tonumber(ARGV[4])
local member = ARGV[5]

redis.call(
    'ZREMRANGEBYSCORE',
    key,
    '-inf',
    now - window
)

local entries = redis.call(
    'ZRANGE',
    key,
    0,
    -1,
    'WITHSCORES'
)

local total = 0
local oldest_timestamp = nil

for i = 1, #entries, 2 do
    local entry_member = entries[i]
    local timestamp = tonumber(entries[i + 1])

    local cost = tonumber(
        string.match(entry_member, ':(%d+)$')
    )

    if not cost then
        return redis.error_reply(
            'Invalid token reservation member: ' .. entry_member
        )
    end

    total = total + cost

    if not oldest_timestamp or timestamp < oldest_timestamp then
        oldest_timestamp = timestamp
    end
end

if total + requested > limit then
    local retry_after = 1

    if oldest_timestamp then
        retry_after = math.max(
            1,
            math.ceil(oldest_timestamp + window - now)
        )
    end

    return {0, total, retry_after}
end

redis.call(
    'ZADD',
    key,
    now,
    member .. ':' .. requested
)

redis.call('EXPIRE', key, window + 5)

return {1, total + requested, 0}
"""


class ContextBudget:
    def __init__(self, redis_url=None, session_id=None):
        self.redis = redis.Redis.from_url(
            redis_url or "redis://127.0.0.1:6379/0",
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=5,
        )
        self.redis.ping()
        self.session_id = session_id or str(uuid.uuid4())
        self.rate_key = "moveiq:groq:tpm"

    @staticmethod
    def estimate_tokens(messages, tools=None):
        payload = json.dumps(
            {"messages": messages, "tools": tools or []},
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        # Conservative approximation, not an exact model tokenizer.
        return math.ceil(len(payload) / 3.0) + 150

    @staticmethod
    def trim_history(messages, max_messages=MAX_HISTORY_MESSAGES):
        if len(messages) <= 1:
            return messages

        system = messages[0]
        history = messages[1:]

        # Keep complete user-turn groups, including their tool messages.
        groups = []
        current = []

        for item in history:
            if item.get("role") == "user":
                if current:
                    groups.append(current)
                current = [item]
            elif current:
                current.append(item)

        if current:
            groups.append(current)

        kept = []
        for group in reversed(groups):
            candidate = group + kept
            if len(candidate) > max_messages and kept:
                break
            kept = candidate

        return [system] + kept

    @classmethod
    def trim_to_budget(
        cls,
        messages,
        tools=None,
        max_tokens=REQUEST_BUDGET - COMPLETION_RESERVE,
    ):
        """Remove oldest complete conversation turns until the prompt fits."""

        if not messages or messages[0].get("role") != "system":
            raise ValueError("Conversation must begin with a system message.")

        system = messages[0]
        history = messages[1:]

        groups = []
        current = []

        for item in history:
            if item.get("role") == "user":
                if current:
                    groups.append(current)
                current = [item]
            elif current:
                current.append(item)

        if current:
            groups.append(current)

        def fits(candidate):
            return cls.estimate_tokens(candidate, tools) <= max_tokens

        kept = list(groups)

        while True:
            candidate = [system]
            for group in kept:
                candidate.extend(group)

            if fits(candidate):
                return candidate

            if len(kept) <= 1:
                raise ValueError(
                    "MoveIQ current turn exceeds the available token budget. "
                    "Please shorten the question or narrow the requested details."
                )

            kept.pop(0)
    def reserve(self, estimated_tokens, completion_tokens=COMPLETION_RESERVE):
        if estimated_tokens < 0 or completion_tokens < 0:
            raise ValueError("Token estimates cannot be negative.")

        cost = estimated_tokens + completion_tokens

        if cost > REQUEST_BUDGET:
            raise ValueError(
                f"MoveIQ request is too large ({cost} estimated tokens). "
                "Please repeat the specific details needed for this question."
            )

        while True:
            now = time.time()

            result = self.redis.eval(
                RESERVE_SCRIPT,
                1,
                self.rate_key,
                now,
                ROLLING_WINDOW_SECONDS,
                cost,
                ROLLING_TPM_LIMIT,
                f"{self.session_id}:{uuid.uuid4()}",
            )

            if not isinstance(result, (list, tuple)) or len(result) != 3:
                raise RuntimeError(
                    f"Unexpected Redis limiter response: {result!r}"
                )

            allowed, _total_used, retry_after = result

            if int(allowed) == 1:
                return

            time.sleep(min(max(float(retry_after), 1.0), 60.0))

