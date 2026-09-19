"""Structured, position-aware extraction of package declarations from OCR output."""

import re
from difflib import SequenceMatcher

from field_vocab import GENERIC_PRODUCT_TERMS

NOT_FOUND = "NOT_FOUND"
FOUND = "FOUND"
UNCERTAIN = "UNCERTAIN"

MARKER_RE = re.compile(r"\b(manufactured|marketed|packed|imported|mfd)\b", re.I)
MARKER_WITH_BY_RE = re.compile(
    r"\b(manufactured|marketed|packed|imported|mfd)\W{0,4}by\b", re.I
)
QUANTITY_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(g|gm|gms|grams?|kg|kgs|ml|l|ltr|litres?)\b", re.I)
DATE_RE = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}[/-]\d{2,4}\b|"
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[.\s-]+\d{2,4}\b",
    re.I,
)
PHONE_RE = re.compile(r"\b\d{10}\b")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.I)
CARE_RE = re.compile(r"\b(customer|consumer)\s*care\b|\bhelpline\b|\btoll\s*-?\s*free\b", re.I)
NET_MARKER_RE = re.compile(r"\bnet\s*(?:weight|wt|quantity)\b|\bcontents?\b", re.I)
SERVING_RE = re.compile(
    r"\bserving\s*size\b|\bper\s*serving\b|"
    r"\bper\s*\d+(?:\.\d+)?\s*(?:g|kg|ml|l)\b|"
    r"\bnutrition\b|\benergy\b|\bunit\s*price\b",
    re.I,
)
DATE_MARKER_RE = re.compile(
    r"\b(?:date\w*\s*(?:of\s*)?(?:packing|packaging)|packed\s*on|"
    r"pkd|mfd|mfg|manufactured(?:\s*on)?)\b",
    re.I,
)
ENTITY_TERMS = {"industries", "foods", "private", "pvt", "limited", "ltd", "llp", "lt"}
NUTRITION_TERMS = {"sugar", "protein", "carbohydrate", "carbohydrates"}


def _empty(reason: str) -> dict:
    return {"value": None, "status": NOT_FOUND, "confidence": 0.0,
            "evidence_text": "", "bbox": None, "reason": reason}


def _bbox(words: list[dict]) -> dict | None:
    if not words:
        return None
    left = min(w["left"] for w in words)
    top = min(w["top"] for w in words)
    right = max(w["left"] + w["width"] for w in words)
    bottom = max(w["top"] + w["height"] for w in words)
    return {"left": left, "top": top, "width": right - left, "height": bottom - top}


def _confidence(words: list[dict], adjustment: float = 0.0) -> float:
    if not words:
        return 0.0
    return round(max(0.0, min(100.0, sum(float(w["conf"]) for w in words) / len(words) + adjustment)), 1)


def _result(value: str | None, status: str, words: list[dict], reason: str) -> dict:
    return {"value": value, "status": status, "confidence": _confidence(words),
            "evidence_text": " ".join(w["text"] for w in words), "bbox": _bbox(words),
            "reason": reason}


def _lines(words: list[dict]) -> list[list[dict]]:
    """Group original-coordinate words into approximate reading lines."""
    result = []
    for word in sorted(words, key=lambda w: (w["top"], w["left"])):
        center = word["top"] + word["height"] / 2
        if result:
            prior = result[-1]
            prior_center = sum(w["top"] + w["height"] / 2 for w in prior) / len(prior)
            if abs(center - prior_center) <= max(10, word["height"]):
                prior.append(word)
                continue
        result.append([word])
    return result


def _line_text(line: list[dict]) -> str:
    return " ".join(w["text"] for w in sorted(line, key=lambda w: w["left"]))


def _matching_words(words: list[dict], match: re.Match) -> list[dict]:
    """Map a text-span match back to its contributing word boxes."""
    chosen, offset = [], 0
    for word in sorted(words, key=lambda w: w["left"]):
        start, end = offset, offset + len(word["text"])
        if start < match.end() and end > match.start():
            chosen.append(word)
        offset = end + 1
    return chosen


def _clean(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def _center(word: dict) -> tuple[float, float]:
    return word["left"] + word["width"] / 2, word["top"] + word["height"] / 2


def _distance(first: dict, second: dict) -> float:
    first_x, first_y = _center(first)
    second_x, second_y = _center(second)
    return ((first_x - second_x) ** 2 + (first_y - second_y) ** 2) ** 0.5


def _nearby(words: list[dict], anchors: list[dict], x_distance: int = 350,
            y_distance: int = 90) -> list[dict]:
    if not anchors:
        return []
    result = []
    for word in words:
        x, y = _center(word)
        if any(abs(x - _center(anchor)[0]) <= x_distance and
               abs(y - _center(anchor)[1]) <= y_distance for anchor in anchors):
            result.append(word)
    return sorted(result, key=lambda w: (w["top"], w["left"]))


def _line_for_word(lines: list[list[dict]], target: dict) -> tuple[int, list[dict]]:
    for index, line in enumerate(lines):
        if any(word is target for word in line):
            return index, line
    return -1, []


def _net_markers(words: list[dict], lines: list[list[dict]]) -> list[list[dict]]:
    markers = []
    for line in lines:
        text = _line_text(line)
        for match in NET_MARKER_RE.finditer(text):
            markers.append(_matching_words(line, match))
    # Tiled OCR may place Net and Weight in adjacent reconstructed lines.
    net_words = [word for word in words if _clean(word["text"]) == "net"]
    weight_words = [word for word in words if _clean(word["text"]) in {"weight", "wt", "quantity"}]
    for net_word in net_words:
        for weight_word in weight_words:
            if abs(_center(net_word)[1] - _center(weight_word)[1]) <= 35 and _distance(net_word, weight_word) <= 180:
                markers.append([net_word, weight_word])
    # Real-label OCR may merge the commodity name and net declaration into one
    # token (for example, "BISCUMTSNETWEIGHT"). The original word remains the
    # evidence source; this only recognizes its embedded declaration context.
    for word in words:
        cleaned = _clean(word["text"])
        if any(marker in cleaned for marker in ("netweight", "netwt", "netquantity")):
            markers.append([word])
    return markers


def _marker_distance(candidate_words: list[dict], markers: list[list[dict]]) -> float:
    if not candidate_words or not markers:
        return float("inf")
    return min(_distance(candidate, marker) for candidate in candidate_words
               for group in markers for marker in group)


def _extract_manufacturer(words: list[dict], lines: list[list[dict]]) -> dict:
    marker_candidates = []
    for index, line in enumerate(lines):
        text = _line_text(line)
        matches = list(MARKER_RE.finditer(text))
        if matches:
            marker_words = []
            for match in matches:
                marker_words.extend(_matching_words(line, match))
            complete = bool(MARKER_WITH_BY_RE.search(text))
            if not complete:
                by_words = [word for word in line if _clean(word["text"]) == "by"]
                complete = any(_distance(marker_word, by_word) <= 120
                               for marker_word in marker_words for by_word in by_words)
            marker_candidates.append((index, marker_words, complete))

    best = None
    for index, marker_words, complete in marker_candidates:
        for line in lines[max(0, index - 1):index + 4]:
            ordered = sorted(line, key=lambda word: word["left"])
            suffixes = [position for position, word in enumerate(ordered)
                        if _clean(word["text"]) in ENTITY_TERMS]
            for suffix in suffixes:
                start = max(0, suffix - 7)
                phrase = ordered[start:suffix + 1]
                # Discard marker verbs and leading address labels from the company value.
                phrase = [word for word in phrase if _clean(word["text"]) not in
                          {"manufactured", "marketed", "packed", "imported", "mfd", "by"}]
                phrase = [word for word in phrase if _clean(word["text"]) not in
                          {"of", "the", "lot", "plot", "no", "number"}
                          and not _clean(word["text"]).isdigit()]
                # OCR line grouping can join two printed columns. A colon is a
                # reliable boundary between a nutrition/table label and the
                # company phrase that follows it.
                colon_positions = [position for position, word in enumerate(phrase[:-1])
                                   if ":" in word["text"]]
                if colon_positions:
                    phrase = phrase[colon_positions[-1] + 1:]
                while phrase and not re.search(r"[a-z0-9]", phrase[0]["text"], re.I):
                    phrase.pop(0)
                while phrase and _clean(phrase[0]["text"]) in {"unit", "unit2", "plot", "no", "number"}:
                    phrase.pop(0)
                if not phrase:
                    continue
                score = _confidence(phrase) + (25 if complete else 0) - min(30, _marker_distance(phrase, [marker_words]) / 20)
                candidate = (score, phrase, marker_words, complete)
                if best is None or candidate[0] > best[0]:
                    best = candidate

    if best:
        _, company_words, marker_words, complete = best
        value = " ".join(word["text"] for word in company_words)
        evidence = sorted({id(word): word for word in marker_words + company_words}.values(),
                          key=lambda word: (word["top"], word["left"]))
        status = FOUND if complete else UNCERTAIN
        reason = ("Selected nearby company/entity text following a complete manufacturer/packer marker."
                  if complete else
                  "Selected nearby plausible company/entity text; the manufacturer/packer marker is incomplete, so inspector review is required.")
        return _result(value, status, evidence, reason)
    if marker_candidates:
        _, marker_words, _ = max(marker_candidates, key=lambda item: _confidence(item[1]))
        return _result(None, UNCERTAIN, marker_words,
                       "A manufacturer/packer marker was detected, but no reliable nearby company/entity phrase was recovered.")
    return _empty("No manufactured, marketed, packed, imported, or mfd marker was detected.")


def _extract_product(words: list[dict], lines: list[list[dict]]) -> dict:
    terms = {term.casefold() for term in GENERIC_PRODUCT_TERMS}
    occurrences = {}
    for word in words:
        cleaned = re.sub(r"[^a-z]", "", word["text"].casefold())
        candidate = cleaned if cleaned in terms else None
        if candidate is None:
            # Match only an OCR-like spelling of a configured commodity term.
            # Split an embedded declaration suffix first; this avoids treating
            # arbitrary long OCR noise as a product name.
            segment = re.split(r"net(?:weight|wt|quantity)|contents?", cleaned)[0]
            close_terms = [term for term in terms
                           if len(segment) >= 4 and abs(len(segment) - len(term)) <= 2
                           and SequenceMatcher(None, segment, term).ratio() >= 0.78]
            if close_terms:
                candidate = max(close_terms, key=lambda term: SequenceMatcher(None, segment, term).ratio())
        if candidate:
            occurrences.setdefault(candidate, []).append([word])
    for term in terms:
        if " " not in term:
            continue
        phrase = re.compile(r"\b" + r"[\s-]+".join(map(re.escape, term.split())) + r"\b", re.I)
        for line in lines:
            for match in phrase.finditer(_line_text(line)):
                evidence = _matching_words(line, match)
                if evidence:
                    occurrences.setdefault(term, []).append(evidence)
    if occurrences:
        heights = sorted(word["height"] for word in words) or [1]
        median_height = heights[len(heights) // 2] or 1
        net_markers = _net_markers(words, lines)
        ranked = []
        for term, match_groups in occurrences.items():
            matches = [word for group in match_groups for word in group]
            best_group = max(match_groups, key=_confidence)
            best_word = max(best_group, key=lambda word: float(word["conf"]))
            index, line = _line_for_word(lines, best_word)
            context_lines = lines[max(0, index - 1):index + 2] if index >= 0 else [[best_word]]
            context = " ".join(_line_text(item) for item in context_lines).casefold()
            candidate_line = _line_text(line).casefold() if line else best_word["text"].casefold()
            score = float(best_word["conf"]) + min(25, best_word["height"] / median_height * 10)
            score += min(24, (len(match_groups) - 1) * 12)
            score += 30 * (len(term.split()) - 1)
            if "ingredient" in context:
                score += 18
            distance = _marker_distance(matches, net_markers)
            if distance < 250:
                score += max(0, 20 - distance / 15)
            nutrition_context = bool(re.search(
                r"nutrition|total\s+sugar|added\s+sugar|protein|carbohydrates?|blood\s+sugar", candidate_line
            ))
            if nutrition_context:
                # A commodity word in a nutrition panel describes a nutrient,
                # not the package's generic product name. Do not substitute a
                # weaker nutrition candidate when no title is visible.
                score -= 120
            else:
                ranked.append((score, term, best_group, len(match_groups)))
        if not ranked:
            return _empty("Configured commodity words were found only in nutrition context, not as a product declaration.")
        _, term, best_evidence, repetitions = max(ranked, key=lambda item: item[0])
        return _result(
            term.upper(), FOUND, best_evidence,
            "Selected the highest-ranked configurable commodity using OCR confidence, prominence, repetition, declaration proximity, and nutrition-context penalties"
            f" ({repetitions} occurrence{'s' if repetitions != 1 else ''}).",
        )
    return _empty("No configurable generic commodity term was detected.")


def _extract_quantity(words: list[dict], lines: list[list[dict]]) -> dict:
    markers = _net_markers(words, lines)
    candidates = []
    for line in lines:
        text = _line_text(line)
        for match in QUANTITY_RE.finditer(text):
            evidence = _matching_words(line, match)
            value = f"{match.group(1)}{match.group(2).lower()}"
            line_index = next((index for index, candidate_line in enumerate(lines) if candidate_line is line), -1)
            nearby_lines = lines[max(0, line_index - 1):line_index + 2] if line_index >= 0 else [line]
            context = " ".join(_line_text(candidate_line) for candidate_line in nearby_lines).casefold()
            distance = _marker_distance(evidence, markers)
            # Per-serving and nutrition-table quantities are measurements, not
            # package net quantity, unless a separate net declaration anchors
            # the value spatially.
            if SERVING_RE.search(context) and distance > 180:
                continue
            score = _confidence(evidence) + (130 if distance <= 180 else 0)
            if SERVING_RE.search(context):
                score -= 160
            candidates.append((score, value, evidence, False, distance))

    # A final 9 can be a lowercase g only when a net declaration anchors it.
    for word in words:
        match = re.fullmatch(r"(\d{2,5})9", word["text"].strip())
        if not match:
            continue
        distance = _marker_distance([word], markers)
        if distance <= 180:
            candidates.append((_confidence([word]) + 130, f"{match.group(1)}g", [word], True, distance))

    if candidates:
        _, value, quantity_words, normalized, distance = max(candidates, key=lambda item: item[0])
        closest_marker = min(markers, key=lambda marker: _marker_distance(quantity_words, [marker])) if markers and distance <= 180 else []
        evidence = sorted({id(word): word for word in closest_marker + quantity_words}.values(),
                          key=lambda word: (word["top"], word["left"]))
        if normalized:
            reason = (f"Normalized OCR token '{quantity_words[0]['text']}' to '{value}' only because it is spatially associated with a Net Weight/Net Quantity marker; original OCR evidence is preserved.")
        elif closest_marker:
            reason = "Selected the highest-ranked quantity candidate because it is associated with a Net Weight/Net Quantity marker and is not serving-size or nutrition context."
        else:
            reason = "Selected the highest-ranked valid quantity candidate after penalizing serving-size, nutrition, energy, and unit-price contexts."
        return _result(value, FOUND, evidence, reason)
    return _empty("No number-and-unit net quantity was detected.")


def _is_price(word: dict) -> bool:
    token = word["text"].strip().replace(",", "")
    return bool(re.fullmatch(r"(?:₹|rs\.?|inr)?\s*\d{1,5}\.\d{2}", token, re.I)) and "/" not in token


def _has_currency_marker(word: dict) -> bool:
    return bool(re.match(r"(?:₹|rs\.?|inr)\s*\d", word["text"].strip(), re.I))


def _is_unit_price(word: dict, line: list[dict]) -> bool:
    token = word["text"].strip().casefold()
    if "/" in token:
        return True
    ordered = sorted(line, key=lambda candidate: candidate["left"])
    try:
        index = next(index for index, candidate in enumerate(ordered) if candidate is word)
    except StopIteration:
        return False
    suffix = " ".join(candidate["text"] for candidate in ordered[index + 1:index + 4])
    return bool(re.match(r"\s*(?:per\b|/)\s*(?:g|kg|ml|l)\b", suffix, re.I))


def _extract_mrp(words: list[dict], lines: list[list[dict]]) -> dict:
    marker_words = [w for w in words if "mrp" in re.sub(r"[^a-z]", "", w["text"].casefold())]
    if not marker_words:
        currency_prices = []
        for word in words:
            if not (_is_price(word) and _has_currency_marker(word)):
                continue
            _, line = _line_for_word(lines, word)
            if not _is_unit_price(word, line) and float(word["conf"]) >= 70:
                currency_prices.append((word, line))
        if not currency_prices:
            return _empty("No MRP marker or reliable non-unit currency price was detected.")
        price, _ = max(currency_prices, key=lambda item: float(item[0]["conf"]))
        return _result(price["text"].strip(), UNCERTAIN, [price],
                       "A non-unit currency price was detected without a reliable MRP marker; it is retained as review evidence and not treated as a legal MRP declaration.")
    candidates = [w for w in words if _is_price(w)]
    candidates = [w for w in candidates if any(
        w["top"] >= marker["top"] - 20 and w["top"] - marker["top"] <= 150
        for marker in marker_words
    )]
    if not candidates:
        return _result(None, UNCERTAIN, marker_words, "MRP marker found, but no nearby reliable price was detected; no price was invented.")
    def price_rank(price: dict) -> tuple[float, float]:
        _, line = _line_for_word(lines, price)
        distance = min(_distance(price, marker) for marker in marker_words)
        return (500.0 if _is_unit_price(price, line) else 0.0, distance)
    non_unit_candidates = [price for price in candidates if price_rank(price)[0] == 0]
    if not non_unit_candidates:
        return _result(None, UNCERTAIN, marker_words,
                       "MRP marker found, but nearby values are unit prices; no legal MRP value was invented.")
    price = min(non_unit_candidates, key=price_rank)
    marker = min(marker_words, key=lambda word: _distance(word, price))
    return _result(price["text"].strip(), FOUND, [marker, price],
                   "Associated an MRP marker variant with the nearest nearby decimal price after excluding unit-price context.")


def _valid_date(value: str) -> bool:
    numeric = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", value)
    if numeric:
        day, month = int(numeric.group(1)), int(numeric.group(2))
        return 1 <= day <= 31 and 1 <= month <= 12
    month_year = re.fullmatch(r"(\d{1,2})[/-](\d{2,4})", value)
    if month_year:
        return 1 <= int(month_year.group(1)) <= 12
    return True


def _date_markers(lines: list[list[dict]]) -> list[dict]:
    markers = []
    for line in lines:
        for match in DATE_MARKER_RE.finditer(_line_text(line)):
            markers.extend(_matching_words(line, match))
    return markers


def _is_full_numeric_date(value: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", value))


def _extract_date(words: list[dict], lines: list[list[dict]]) -> dict:
    matches = []
    for line in lines:
        text = _line_text(line)
        for match in DATE_RE.finditer(text):
            value = match.group(0)
            evidence = _matching_words(line, match)
            if _valid_date(value):
                matches.append((value, evidence))
    marker_evidence = _date_markers(lines)
    if not matches:
        if not marker_evidence:
            date_words = [word for word in words if _clean(word["text"]).startswith("date")]
            packing_words = [word for word in words if _clean(word["text"]) in
                             {"packing", "packaging", "manufacture", "manufacturing"}]
            for date_word in date_words:
                close = [word for word in packing_words if _distance(date_word, word) <= 180]
                if close:
                    marker_evidence.extend([date_word, min(close, key=lambda word: _distance(date_word, word))])
        if marker_evidence:
            return _result(None, UNCERTAIN, marker_evidence,
                           "A packing/manufacture-date marker was detected, but no reliable date value was recovered; no date was reconstructed.")
        return _empty("No supported date or packing/manufacture-date marker was detected.")
    def date_rank(candidate: tuple[str, list[dict]]) -> tuple[int, float, float, int]:
        value, evidence = candidate
        confidence = _confidence(evidence)
        distance = _marker_distance(evidence, [marker_evidence]) if marker_evidence else float("inf")
        reliable = confidence >= 70 and (distance <= 220 or _is_full_numeric_date(value))
        return (1 if reliable else 0, confidence, -distance, len(value))
    value, evidence = max(matches, key=date_rank)
    confidence = _confidence(evidence)
    distance = _marker_distance(evidence, [marker_evidence]) if marker_evidence else float("inf")
    combined_evidence = sorted({id(word): word for word in evidence + marker_evidence}.values(),
                               key=lambda word: (word["top"], word["left"]))
    if confidence >= 70 and (distance <= 220 or _is_full_numeric_date(value)):
        return _result(value, FOUND, combined_evidence,
                       "Selected a calendar-valid date with reliable OCR confidence and packing/manufacture context, or a complete supported numeric date declaration.")
    return _result(None, UNCERTAIN, combined_evidence or evidence,
                   "A date-like OCR fragment was found, but its confidence or packing/manufacture-marker proximity is insufficient; no date was accepted.")


def _extract_consumer_care(words: list[dict], lines: list[list[dict]]) -> dict:
    care_anchors = []
    for line in lines:
        text = _line_text(line)
        care_match = re.search(r"\b(?:customer|consumer|quality)\s*care\b|\bhelpline\b|\btoll\s*-?\s*free\b", text, re.I)
        if care_match:
            care_anchors.extend(_matching_words(line, care_match))
    if not care_anchors:
        care_words = [word for word in words if _clean(word["text"]) == "care"]
        qualifier_words = [word for word in words if _clean(word["text"]) in
                           {"customer", "consumer", "quality", "helpline"}]
        for care_word in care_words:
            close = [word for word in qualifier_words if _distance(care_word, word) <= 120]
            if close:
                care_anchors.extend([min(close, key=lambda word: _distance(care_word, word)), care_word])

    search_words = _nearby(words, care_anchors, x_distance=500, y_distance=90) if care_anchors else words
    for line in _lines(search_words):
        text = _line_text(line)
        for pattern, label in ((EMAIL_RE, "email"), (PHONE_RE, "phone number")):
            match = pattern.search(text)
            if match:
                evidence = care_anchors + _matching_words(line, match)
                return _result(match.group(0), FOUND, evidence, f"Matched a valid consumer-care {label} near care/helpline evidence.")
        ordered = sorted(line, key=lambda word: word["left"])
        for start, word in enumerate(ordered):
            if not re.fullmatch(r"\d{1,4}", word["text"].strip()):
                continue
            fragments, digits = [], ""
            for candidate in ordered[start:]:
                token = candidate["text"].strip()
                if not re.fullmatch(r"\d{1,4}", token):
                    break
                fragments.append(candidate)
                digits += token
                if len(digits) >= 10:
                    break
            if len(digits) == 10:
                evidence = care_anchors + fragments
                return _result(digits, FOUND, evidence,
                               "Matched a ten-digit consumer-care phone number from contiguous OCR digit fragments near care/helpline evidence.")
    if care_anchors:
        fragments = [word for word in search_words if
                     min(abs(_center(word)[1] - _center(anchor)[1]) for anchor in care_anchors) <= 35
                     and ("@" in word["text"] or
                          re.fullmatch(r"[\d\s()+,.-]{4,}", word["text"].strip()))]
        evidence = sorted({id(word): word for word in care_anchors + fragments}.values(),
                          key=lambda word: (word["top"], word["left"]))
        return _result(None, UNCERTAIN, evidence,
                       "Consumer/quality-care wording was detected with incomplete or corrupted contact fragments; no phone number or email was repaired or invented.")
    return _empty("No customer/consumer-care marker, ten-digit phone number, email, or helpline was detected.")


def extract_fields(ocr_result: dict) -> dict:
    """Return field records without changing any existing checker/API response."""
    words = ocr_result.get("words", [])
    lines = _lines(words)
    return {
        "manufacturer_or_packer": _extract_manufacturer(words, lines),
        "generic_product_name": _extract_product(words, lines),
        "net_quantity": _extract_quantity(words, lines),
        "mrp": _extract_mrp(words, lines),
        "manufacture_or_packing_date": _extract_date(words, lines),
        "consumer_care": _extract_consumer_care(words, lines),
    }
