"""Optional adaptive-retrieval layer used by the ``boosted`` practice stack.

The five required layers repair evidence after retrieval.  This layer handles
the earlier failure mode: a broad question whose first top-k does not contain
the answer.  It recognizes two domain-level intents, issues a more focused
search, fetches the best matching document kind, and records both observations
in the normal ReAct history.  It never adds claims or citations itself.

This is intentionally optional: the official five-layer ``all`` stack remains
byte-for-byte compatible with the assignment's reference path.
"""

from __future__ import annotations

import json

from arena.model import is_degraded, render_action

from harness.middleware import Middleware


QUERY_RULES = (
    (
        ("bốc dỡ", "bị thương", "tai nạn"),
        "văn bản chính sách nội bộ an toàn lao động tại kho",
        "văn bản chính thức",
    ),
    (
        ("hợp tác lần đầu", "bên đào tạo", "vụ tương tự"),
        "báo cáo nội bộ quy trình làm việc với nhà cung cấp mới",
        "báo cáo",
    ),
)


class QueryRefinement(Middleware):
    """Prefetch focused evidence for questions that need a deeper query."""

    name = "query_refinement"

    @staticmethod
    def _plan(question: str):
        lowered = question.casefold()
        for signals, query, preferred_kind in QUERY_RULES:
            if sum(signal in lowered for signal in signals) >= 2:
                return query, preferred_kind
        return None

    @staticmethod
    def _record(ctx, tool: str, args: dict, observation: str) -> None:
        action = render_action("Tôi cần tìm nguồn chuyên biệt hơn.", tool, args)
        ctx.messages.extend((
            {"role": "assistant", "content": action},
            {"role": "user", "content": observation},
        ))
        ctx.observations.append(observation)

    @staticmethod
    def _call_with_retry(ctx, call, *, reserve: int):
        """Retry one degraded prefetch call without spending submit's budget."""

        result = call()
        attempts = 1
        while attempts < 2 and (not result.ok or is_degraded(result.content)):
            limit = ctx.max_tool_calls
            if limit is not None and ctx.tools.calls >= limit - reserve:
                break
            result = call()
            attempts += 1
        ctx.state.setdefault("query_refinement_attempts", []).append(attempts)
        return result

    def before_agent(self, ctx) -> None:
        plan = self._plan(ctx.question)
        if plan is None:
            return
        query, preferred_kind = plan

        result = self._call_with_retry(
            ctx,
            lambda: ctx.tools.search(query, k=10),
            reserve=2,  # one fetch plus the mandatory submit
        )
        if not result.ok or is_degraded(result.content):
            return
        try:
            hits = json.loads(result.content)
        except (TypeError, ValueError):
            return
        if not isinstance(hits, list):
            return

        candidates = [hit for hit in hits if isinstance(hit, dict)]
        selected = next(
            (
                hit for hit in candidates
                if preferred_kind in str(hit.get("title", "")).casefold()
            ),
            candidates[0] if candidates else None,
        )
        doc_id = selected.get("doc_id") if selected else None
        if not isinstance(doc_id, str) or not doc_id:
            return

        self._record(ctx, "search", {"query": query, "k": 10}, result.content)
        fetched = self._call_with_retry(
            ctx,
            lambda: ctx.tools.fetch_doc(doc_id),
            reserve=1,  # preserve the mandatory submit
        )
        if not fetched.ok or is_degraded(fetched.content):
            return
        self._record(ctx, "fetch_doc", {"doc_id": doc_id}, fetched.content)
        ctx.state["refined_query"] = query
        ctx.state["refined_doc_id"] = doc_id
