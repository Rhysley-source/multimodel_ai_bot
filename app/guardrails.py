"""
Guardrails — input and output safety checks for the chat API.

Layers:
  1. Rate limiting     — per-user request cap (in-memory sliding window)
  2. Input checks      — prompt injection detection, blocked topics
  3. Output checks     — system prompt leakage detection
"""
import logging
import re
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


# ── 1. Rate limiter ────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Sliding-window rate limiter keyed by user ID.
    Allows max_requests within window_seconds.
    Thread-safe for asyncio (single-threaded event loop).
    """

    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._buckets: dict[int, deque] = defaultdict(deque)

    def is_allowed(self, user_id: int) -> bool:
        now = time.monotonic()
        bucket = self._buckets[user_id]

        # Drop timestamps outside the window
        while bucket and now - bucket[0] > self.window:
            bucket.popleft()

        if len(bucket) >= self.max_requests:
            logger.warning(
                "Rate limit exceeded | user_id=%s requests=%d window=%ds",
                user_id, len(bucket), self.window,
            )
            return False

        bucket.append(now)
        return True

    def remaining(self, user_id: int) -> int:
        now = time.monotonic()
        bucket = self._buckets[user_id]
        while bucket and now - bucket[0] > self.window:
            bucket.popleft()
        return max(0, self.max_requests - len(bucket))


# One shared instance — 20 chat messages per user per minute
chat_rate_limiter = RateLimiter(max_requests=20, window_seconds=60)


# ── 2. Input guardrails ────────────────────────────────────────────────────────

# Prompt injection patterns — attempt to override or ignore the system prompt
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|your)\s+(instructions?|prompts?|rules?|guidelines?|context)",
    r"disregard\s+(your|all|previous)\s+(instructions?|prompts?|system\s+prompt)",
    r"forget\s+(everything|all|your\s+instructions|your\s+prompt)",
    r"override\s+(your\s+)?(safety|restrictions?|guidelines?|rules?)",
    r"you\s+are\s+now\s+(dan|an?\s+AI\s+without|a\s+different)",
    r"pretend\s+(you\s+)?(are|have\s+no|don'?t\s+have)\s+(restrictions?|guidelines?|rules?|ethics?)",
    r"act\s+as\s+(if\s+you\s+(are|were|have\s+no)|a\s+version\s+of)",
    r"(jailbreak|dan\s+mode|developer\s+mode|sudo\s+mode|god\s+mode)",
    r"do\s+anything\s+now",
    r"(reveal|show|print|output|display|tell\s+me)\s+(your\s+)?(system\s+prompt|instructions|guidelines|rules)",
]

_COMPILED_INJECTION = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# Blocked topic keywords (expand based on your use case)
_BLOCKED_KEYWORDS = [
    r"\b(how\s+to\s+make\s+(a\s+)?(bomb|weapon|explosive|poison|malware|virus|ransomware))\b",
    r"\b(child\s+(pornography|abuse|exploitation))\b",
    r"\b(detailed\s+(instructions?|steps?|guide)\s+(to|for)\s+(kill|murder|harm|attack))\b",
]

_COMPILED_BLOCKED = [re.compile(p, re.IGNORECASE) for p in _BLOCKED_KEYWORDS]


def check_input(message: str, user_id: int) -> tuple[bool, str]:
    """
    Validate user input before sending to LLM.
    Returns (is_safe, reason). is_safe=True means the message passed all checks.
    """
    # Prompt injection check
    for pattern in _COMPILED_INJECTION:
        if pattern.search(message):
            logger.warning(
                "Prompt injection attempt | user_id=%s pattern=%s",
                user_id, pattern.pattern[:60],
            )
            return False, "Your message contains content that violates our usage policy."

    # Blocked content check
    for pattern in _COMPILED_BLOCKED:
        if pattern.search(message):
            logger.warning(
                "Blocked content detected | user_id=%s pattern=%s",
                user_id, pattern.pattern[:60],
            )
            return False, "Your message contains content that is not permitted."

    return True, ""


# ── 3. Output guardrails ───────────────────────────────────────────────────────

# Patterns that suggest the LLM leaked the system prompt in its response
_LEAKAGE_PATTERNS = [
    r"\[IDENTITY\s*[—-]\s*follow strictly\]",
    r"\[TASK\]",
    r"system\s+prompt\s+(is|says|reads|states)[\s:]+",
]

_COMPILED_LEAKAGE = [re.compile(p, re.IGNORECASE) for p in _LEAKAGE_PATTERNS]


def check_output(response: str, model: str, user_id: int) -> tuple[bool, str]:
    """
    Validate LLM response before returning to the user.
    Returns (is_safe, reason). is_safe=True means the response is clean.
    """
    for pattern in _COMPILED_LEAKAGE:
        if pattern.search(response):
            logger.warning(
                "System prompt leakage in response | user_id=%s model=%s pattern=%s",
                user_id, model, pattern.pattern[:60],
            )
            return False, "Response was filtered for security reasons."

    return True, ""
