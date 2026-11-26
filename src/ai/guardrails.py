import re


# Numeric fleet and staffing prescriptions.
OPERATIONAL_QUANTITY_PATTERNS = [
    # Percentage-based operational recommendations.
    re.compile(
        r"\b\d+(?:\.\d+)?\s*%\s*"
        r"(?:more|less|fewer|additional|extra|increase|decrease|"
        r"above|below|of)\b.{0,70}\b"
        r"(?:vehicles?|taxis?|cabs?|fleet|drivers?|staff|workforce)\b",
        re.IGNORECASE,
    ),

    # Action followed by a quantity, optionally with a modifier.
    # Examples: allocate 25 additional taxis; schedule 15 fewer drivers.
    re.compile(
        r"\b(?:deploy|allocate|add|remove|reduce|increase|"
        r"schedule|assign|keep|maintain)\b.{0,45}"
        r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*"
        r"(?:(?:more|less|fewer|additional|extra)\s+)?"
        r"(?:vehicles?|taxis?|cabs?|drivers?|staff|workforce)\b",
        re.IGNORECASE,
    ),

    # Quantity and modifier followed by an operational resource.
    # Examples: 25 additional taxis; 15 fewer drivers.
    re.compile(
        r"\b\d+(?:,\d{3})*(?:\.\d+)?\s+"
        r"(?:more|less|fewer|additional|extra)\s+"
        r"(?:vehicles?|taxis?|cabs?|drivers?|staff|workforce)\b",
        re.IGNORECASE,
    ),

    # Quantity followed by an operational resource and an action.
    # Examples: 25 taxis to deploy; 15 drivers to remove.
    re.compile(
        r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*"
        r"(?:vehicles?|taxis?|cabs?|drivers?|staff)\b.{0,35}"
        r"\b(?:more|less|fewer|additional|extra|increase|reduce|"
        r"deploy|allocate|assign|remove)\b",
        re.IGNORECASE,
    ),
]

# Unsupported causal explanations for mobility demand.
CAUSAL_CLAIM_PATTERNS = [
    re.compile(
        r"\b(?:driven by|caused by|attributable to|"
        r"resulting from|a result of|due to|because of)\s+"
        r"(?:the\s+)?"
        r"(?:nightlife|events?|concerts?|commuting|"
        r"post[- ]work|airport flights?|weather|traffic|"
        r"tourism|business activity|rush hour)\b",
        re.IGNORECASE,
    ),
]

# Unsupported financial projections.
FINANCIAL_PROJECTION_PATTERNS = [
    re.compile(
        r"\b(?:will|would|could)\s+(?:generate|save|earn|"
        r"produce|increase|decrease|reduce)\b.{0,65}"
        r"\b(?:revenue|profit|savings|costs?|income)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:increase|decrease|improve|reduce)\b.{0,45}"
        r"\b(?:revenue|profit|savings|costs?|income)\b.{0,35}"
        r"\bby\s+\$?\d",
        re.IGNORECASE,
    ),
]


def validate_response(answer):
    """
    Conservative risk-based response validation.

    This checks known high-risk unsupported claim patterns.
    It is not a complete numerical fact-checker.
    """
    if not isinstance(answer, str) or not answer.strip():
        return {
            "passed": False,
            "violations": ["empty_response"],
        }

    violations = []

    pattern_groups = [
        (
            "unsupported_fleet_or_staffing_quantity",
            OPERATIONAL_QUANTITY_PATTERNS,
        ),
        (
            "unsupported_causal_explanation",
            CAUSAL_CLAIM_PATTERNS,
        ),
        (
            "unsupported_financial_projection",
            FINANCIAL_PROJECTION_PATTERNS,
        ),
    ]

    for violation_name, patterns in pattern_groups:
        if any(pattern.search(answer) for pattern in patterns):
            violations.append(violation_name)

    return {
        "passed": not violations,
        "violations": violations,
    }


def safe_fallback(violations):
    """Return a transparent response when a risk check fails."""
    return (
        "I retrieved the relevant MoveIQ data, but I could not validate "
        "all parts of the generated response. I have withheld the "
        "unsupported claims rather than present them as facts.\n\n"
        "The available forecast can identify predicted demand peaks and "
        "quiet periods. However, exact vehicle allocations, staffing "
        "requirements, financial impact, and the causes of demand changes "
        "cannot be established without additional operational data."
    )
