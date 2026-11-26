import re
from decimal import Decimal, InvalidOperation


# Strictly formatted numbers: accepts 1,234.56 but not malformed 1,
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
    r"(?=$|[^A-Za-z0-9_]|[kKmMbBtT](?![A-Za-z]))"
)

METRIC_FIELDS = {
    "trips": {
        "total_trips", "trip_count", "pickup_count",
        "valid_fare_trips", "valid_fare_records",
        "valid_distance_records", "valid_duration_records",
    },
    "revenue": {
        "fare_revenue", "revenue", "total_revenue",
    },
    "average_fare": {
        "average_fare", "avg_fare",
    },
    "distance": {
        "average_distance_miles", "avg_distance",
        "average_distance", "valid_distance_records",
    },
    "duration": {
        "average_duration_minutes", "avg_duration",
        "average_duration", "valid_duration_records",
    },
    "predicted_demand": {
        "predicted_demand",
    },
    "fare": {
        "fare", "fare_amount", "valid_fare_records",
    },
}

METRIC_TERMS = {
    "average_fare": re.compile(
        r"\b(avg|average|mean)\s+(fare|price)\b|\baverage_fare\b",
        re.I,
    ),
    "predicted_demand": re.compile(
        r"\b(predicted|forecast|projected)\s+"
        r"(demand|trips|pickups)\b|\bpredicted_demand\b",
        re.I,
    ),
    "revenue": re.compile(
        r"\b(revenue|earnings|fare\s+revenue)\b",
        re.I,
    ),
    "distance": re.compile(
        r"\b(distance|miles|mileage)\b",
        re.I,
    ),
    "duration": re.compile(
        r"\b(duration|trip\s+time|minutes\s+per\s+trip)\b",
        re.I,
    ),
    "trips": re.compile(
        r"\b(trips|pickups|trip\s+count|number\s+of\s+trips)\b",
        re.I,
    ),
    "fare": re.compile(
        r"\b(fare|fares|fare\s+amount)\b",
        re.I,
    ),
}

ENTITY_FIELDS = {
    "borough", "zone", "zone_name", "pickup_zone",
    "dropoff_zone", "segment", "name",
}

MONTH_PATTERN = (
    r"January|February|March|April|May|June|July|August|"
    r"September|October|November|December|Jan|Feb|Mar|Apr|"
    r"Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)


def _number(value):
    """Convert a numeric value or formatted numeric string to Decimal."""
    if isinstance(value, bool) or value is None:
        return None

    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None

    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("$", "")
        cleaned = cleaned.replace("%", "")
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None

    return None



def _matches_scaled_claim(text, match, values):
    """Match scaled quantities using the precision shown in the answer."""
    claim = _number(match.group())
    if claim is None:
        return False

    suffix = text[match.end():]
    scale_match = re.match(
        r"\s*(thousand|million|billion|trillion)\b"
        r"|([kmbt])(?![A-Za-z])",
        suffix,
        re.I,
    )

    if not scale_match:
        return _matches(claim, values)

    unit = (scale_match.group(1) or scale_match.group(2)).lower()
    scales = {
        "thousand": Decimal("1000"),
        "k": Decimal("1000"),
        "million": Decimal("1000000"),
        "m": Decimal("1000000"),
        "billion": Decimal("1000000000"),
        "b": Decimal("1000000000"),
        "trillion": Decimal("1000000000000"),
        "t": Decimal("1000000000000"),
    }
    multiplier = scales[unit]
    scaled_claim = claim * multiplier

    # Example: 3.48 million represents rounding to the nearest 10,000.
    quantum = Decimal("1").scaleb(claim.as_tuple().exponent) * multiplier

    for value in values:
        if (value / quantum).quantize(Decimal('1')) * quantum == scaled_claim:
            return True

    return False


def _remove_contextual_dates_and_times(text):
    """Remove contextual date/time and clearly labelled ranking metadata."""

    # Join Unicode spaces inside thousands-separated numbers.
    text = re.sub(
        r"(?<=\d)[\u00a0\u202f\u2009](?=\d{3}(?:\D|$))",
        "",
        text,
    )
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = text.replace("\u2009", " ")
    text = text.replace("\u2011", "-").replace("\u2010", "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    # Ignore Markdown ordered-list markers, not numbers in the list content.
    text = re.sub(
        r"(?m)^[ \t]{0,3}\d{1,3}[.)][ \t]+(?=\S)",
        "",
        text,
    )

    # Remove the first-column number only in tables with an explicit Rank header.
    lines = text.splitlines(keepends=True)
    cleaned_lines = []
    rank_table = False

    for line in lines:
        if re.search(r"\|\s*Rank\s*\|", line, re.I):
            rank_table = True
            cleaned_lines.append(line)
            continue

        if rank_table and "|" in line:
            line = re.sub(
                r"^(\s*\|)\s*\d+\s*(\|)",
                r"\1 \2",
                line,
                count=1,
            )
        elif line.strip() and "|" not in line:
            rank_table = False

        cleaned_lines.append(line)

    text = "".join(cleaned_lines)

    # Remove counts only when explicitly used as top/bottom ranking labels.
    text = re.sub(
        r"\b(?:top|bottom)\s*-\s*\d+\b",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\b(?:top|bottom)\s+\d+\b",
        " ",
        text,
        flags=re.I,
    )

    # Remove a duration only on a line explicitly labelled Forecast horizon.
    text = re.sub(
        r"(?im)^([^\n]*\bforecast\s+horizon\b[^\n]*?)\b\d+\s*hours?\b",
        r"\1",
        text,
    )

    # Remove numbered labels such as #1, not ordinary standalone numbers.
    text = re.sub(r"#\s*\d+\b", " ", text)

    months = (
        "January|February|March|April|May|June|July|August|"
        "September|October|November|December|Jan|Feb|Mar|Apr|"
        "Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
    )

    # Month-name date ranges.
    text = re.sub(
        r"\b(?:" + months + r")\s+\d{1,2}\s*-\s*"
        r"(?:(?:" + months + r")\s+)?\d{1,2}[,]?\s+\d{4}\b",
        " ",
        text,
        flags=re.I,
    )

    # ISO timestamps and dates.
    text = re.sub(
        r"\b\d{4}-\d{1,2}-\d{1,2}\s+"
        r"(?:[01]?\d|2[0-3]):[0-5]\d"
        r"(?::[0-5]\d(?:\.\d+)?)?(?:Z)?\b",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\b\d{4}-\d{1,2}-\d{1,2}\b", " ", text)

    # Numeric dates.
    text = re.sub(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        " ",
        text,
    )

    # Month-name dates and month/year labels.
    text = re.sub(
        r"\b(?:" + months + r")\s+\d{1,2}"
        r"(?:st|nd|rd|th)?[,]?(?:\s+\d{4})?\b",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\b(?:" + months + r")\s+\d{4}\b",
        " ",
        text,
        flags=re.I,
    )

    # Clock times.
    text = re.sub(
        r"\b(?:[01]?\d|2[0-3]):[0-5]\d"
        r"(?::[0-5]\d)?\s*(?:AM|PM)?\b",
        " ",
        text,
        flags=re.I,
    )

    return text

def _iter_records(value, inherited_entities=None):
    """
    Recursively yield dictionary records with entity labels inherited
    from their parent dictionaries.
    """
    if inherited_entities is None:
        inherited_entities = {}

    if isinstance(value, dict):
        current_entities = dict(inherited_entities)

        for key, item in value.items():
            if key.lower() in ENTITY_FIELDS and isinstance(item, (str, int)):
                current_entities[key.lower()] = str(item)

        yield value, current_entities

        for item in value.values():
            if isinstance(item, (dict, list)):
                yield from _iter_records(item, current_entities)

    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (dict, list)):
                yield from _iter_records(item, inherited_entities)


def _find_mentioned_entities(sentence, results):
    """Find result-backed entity labels explicitly mentioned in a sentence."""
    sentence_lower = sentence.lower()
    found = set()

    for result in results or []:
        if not isinstance(result, dict):
            continue

        for record, entities in _iter_records(result.get("data", {})):
            for value in entities.values():
                label = str(value).strip()
                if len(label) >= 2 and label.lower() in sentence_lower:
                    found.add(label.lower())

    return found


def _collect_metric_values(results, metric, entity_labels=None):
    """Collect values only from fields belonging to the requested metric."""
    fields = METRIC_FIELDS.get(metric, set())
    values = []

    for result in results or []:
        if not isinstance(result, dict):
            continue

        for record, entities in _iter_records(result.get("data", {})):
            if entity_labels:
                record_labels = {
                    str(value).lower()
                    for value in entities.values()
                }

                # Do not use unlabelled records to support an entity-specific claim.
                if not record_labels.intersection(entity_labels):
                    continue

            for key, value in record.items():
                if key.lower() in fields:
                    number = _number(value)
                    if number is not None:
                        values.append(number)

    return values


def _collect_all_numbers(results):
    """Collect numeric evidence from tool results as a fallback."""
    values = []

    def visit(item):
        if isinstance(item, dict):
            for value in item.values():
                visit(value)
        elif isinstance(item, list):
            for value in item:
                visit(value)
        elif isinstance(item, (int, float, Decimal)) and not isinstance(item, bool):
            number = _number(item)
            if number is not None:
                values.append(number)
        elif isinstance(item, str):
            for match in NUMBER_PATTERN.finditer(
                _remove_contextual_dates_and_times(item)
            ):
                number = _number(match.group())
                if number is not None:
                    values.append(number)

    for result in results or []:
        if isinstance(result, dict):
            visit(result.get("data", {}))

    return values


def _matches(claim, values):
    """Check exact values and reasonable display rounding."""
    if claim is None:
        return False

    for value in values:
        if claim == value:
            return True

        # Permit ordinary display rounding to two decimal places.
        if claim.quantize(Decimal("0.01")) == value.quantize(Decimal("0.01")):
            return True

        # Permit integer display of a source value.
        if claim == value.quantize(Decimal("1")):
            return True

    return False


def _matches_percentage(claim, values):
    """Match percentages using only the precision displayed in the claim."""
    if claim is None:
        return False

    quantum = Decimal("1").scaleb(claim.as_tuple().exponent)

    for value in values:
        if claim.quantize(quantum) == value.quantize(quantum):
            return True

    return False


def _nearest_metric(sentence, number_start, number_end):
    """Identify the metric phrase closest to an individual numeric claim."""
    candidates = []

    for metric, pattern in METRIC_TERMS.items():
        for match in pattern.finditer(sentence):
            if match.end() <= number_start:
                distance = number_start - match.end()
            elif match.start() >= number_end:
                distance = match.start() - number_end
            else:
                distance = 0

            candidates.append((distance, metric))

    if not candidates:
        return None

    return min(candidates, key=lambda item: item[0])[1]


def _sentence_metric(sentence):
    """Identify the metric explicitly discussed in a sentence."""
    for metric in (
        "average_fare",
        "predicted_demand",
        "revenue",
        "distance",
        "duration",
        "trips",
        "fare",
    ):
        if METRIC_TERMS[metric].search(sentence):
            return metric

    return None


def _percentage_changes(results):
    """
    Calculate percentage changes for comparable numeric fields in
    two-record segment comparisons.
    """
    changes = []

    def inspect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, list) and len(item) == 2:
                    first, second = item

                    if isinstance(first, dict) and isinstance(second, dict):
                        calculate(first, second)

                inspect(item)

        elif isinstance(value, list):
            if len(value) == 2 and all(isinstance(x, dict) for x in value):
                calculate(value[0], value[1])

            for item in value:
                inspect(item)

    def calculate(first, second):
        for key in first.keys() & second.keys():
            if key.lower() in ENTITY_FIELDS:
                continue

            a = _number(first[key])
            b = _number(second[key])

            if a is None or b is None or a == 0:
                continue

            changes.append(((b - a) / abs(a)) * Decimal("100"))

    for result in results or []:
        if not isinstance(result, dict):
            continue

        if result.get("tool") == "compare_segments":
            inspect(result.get("data", {}))

    return changes


def validate_numeric_evidence(answer, tool_results=None):
    """
    Validate numeric claims against successful tool-result evidence.

    Returns:
        {
            "passed": bool,
            "unsupported_numbers": list,
            "reason": str
        }
    """
    results = [
        result
        for result in (tool_results or [])
        if isinstance(result, dict)
        and isinstance(result.get("data"), (dict, list))
        and not result.get("error")
    ]

    if not isinstance(answer, str) or not answer.strip():
        return {
            "passed": True,
            "unsupported_numbers": [],
            "reason": "No numeric answer to validate.",
        }

    unsupported = []
    all_values = _collect_all_numbers(results)
    percentage_values = _percentage_changes(results)

    # Remove Markdown ordered-list markers before sentence splitting.
    # This excludes list indices while preserving numbers in list content.
    answer_for_audit = re.sub(
        r"(?m)^[ \t]{0,3}\d{1,3}[.)][ \t]+(?=\S)",
        "",
        answer.strip(),
    )

    # Split into sentence-like units while retaining the sentence context.
    sentences = re.split(r"(?<=[.!?])\s+", answer_for_audit)

    for sentence in sentences:
        # Validate comparison clauses independently so one entity cannot
        # borrow another entity's numeric evidence.
        clauses = re.split(
            r"\s+\b(?:while|whereas|but)\b\s+",
            sentence,
            flags=re.IGNORECASE,
        )

        for clause in clauses:
            numeric_text = _remove_contextual_dates_and_times(clause)
            matches = list(NUMBER_PATTERN.finditer(numeric_text))

            if not matches:
                continue

            entities = _find_mentioned_entities(clause, results)

            for match in matches:
                raw = match.group()
                claim = _number(raw)

                if claim is None:
                    continue

                if raw.endswith("%"):
                    supported = _matches_percentage(claim, percentage_values)
                else:
                    metric = (
                        _nearest_metric(numeric_text, match.start(), match.end())
                        or _sentence_metric(clause)
                        or _sentence_metric(sentence)
                    )
                    has_forecast_evidence = any(
                        isinstance(result, dict)
                        and result.get("tool") == "get_forecast_insights"
                        for result in results
                    )
                    if (
                        has_forecast_evidence
                        and re.search(r"\b(busiest|quietest)\b", clause, re.I)
                    ):
                        metric = "predicted_demand"

                    if metric:
                        evidence = _collect_metric_values(
                            results,
                            metric,
                            entities if entities else None,
                        )
                    else:
                        evidence = all_values

                    supported = _matches_scaled_claim(numeric_text, match, evidence)
                    if (
                        not supported
                        and metric == "predicted_demand"
                        and claim == claim.to_integral_value()
                    ):
                        supported = any(abs(claim - value) < Decimal("1") for value in evidence)

                if not supported:
                    unsupported.append(raw)

    return {
        "passed": len(unsupported) == 0,
        "unsupported_numbers": unsupported,
        "reason": (
            "All numeric claims matched available tool evidence."
            if not unsupported
            else "One or more numeric claims could not be matched to tool evidence."
        ),
    }
