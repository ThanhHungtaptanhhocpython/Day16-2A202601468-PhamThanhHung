"""A stronger offline model used only by the optional ``boosted`` practice path.

The frozen :class:`arena.model.MockModel` deliberately fuses contradictory
sources and never emits the optional synthesis ``verdict`` field.  Middleware
cannot legally reconstruct either value because the scorer requires the text
to originate in a model FINAL.  This subclass changes only those two mock
answering behaviours; it does not read brief ids, required facts, or answers.
"""

from __future__ import annotations

import re

from arena.model import MockModel, _first_user_content


_OPTION_RE = re.compile(
    r"\([a-z]\)\s*(.+?)(?=;\s*\([a-z]\)|[.!?]\s*$)",
    re.IGNORECASE,
)
_UNCERTAINTY_MARKERS = ("chưa đủ", "không đủ")
_NON_DECISIVE_EVIDENCE = (
    "chỉ mang tính tổng hợp",
    "không thay thế cho văn bản chính sách",
)


class BoostedMockModel(MockModel):
    """Preserve full conflict evidence and answer closed synthesis choices."""

    def __init__(self, corpus, seed: int) -> None:
        super().__init__(corpus=corpus, seed=seed)
        self._current_question = ""

    def complete(self, messages: list[dict], **kw):
        self._current_question = _first_user_content(messages)
        return super().complete(messages, **kw)

    def _final_payload(self, evidence_ids: list[str], conversation: str) -> dict:
        payload = super()._final_payload(evidence_ids, conversation)
        usable = self._usable_evidence(evidence_ids, conversation)

        conflicting = [
            (doc, span) for doc, span in usable if "contradiction" in doc.tags
        ]
        if len(conflicting) >= 2:
            claims = [
                {"text": span, "doc_id": doc.doc_id}
                for doc, span in conflicting[:2]
            ]
            payload.update(
                answer=(
                    "Hai nguồn nội bộ đang mâu thuẫn; cần xác minh thẩm quyền "
                    "trước khi áp dụng: " + " ".join(claim["text"] for claim in claims)
                ),
                citations=[claim["doc_id"] for claim in claims],
                abstain=True,
                claims=claims,
            )

        options = [
            option.strip() for option in _OPTION_RE.findall(self._current_question)
        ]
        evidence = conversation.casefold()
        non_decisive = any(marker in evidence for marker in _NON_DECISIVE_EVIDENCE)
        if options and non_decisive:
            verdict = next(
                (
                    option
                    for option in options
                    if any(marker in option.casefold() for marker in _UNCERTAINTY_MARKERS)
                ),
                None,
            )
            if verdict is not None:
                payload["verdict"] = verdict

        return payload


__all__ = ["BoostedMockModel"]
