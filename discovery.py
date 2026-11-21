"""`dynamic_delegation_models` — discovery and dry-run for the `route` parameter.

Core never validates a delegation route's model id, so the agent needs a way to see what is real
and to test a spec before spending a subagent on it. This tool answers from the same catalog the
provider resolves against, and never blocks on the network.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from . import catalog as cat
from .provider import get_catalog, refresh_catalog

TOOLSET = "delegation"  # core toolset: visible wherever delegate_task is

SCHEMA = {
    "name": "dynamic_delegation_models",
    "description": (
        "List, search and dry-run the provider/model/thinking-level specs accepted by "
        "delegate_task's `route` parameter. Use it before delegating: core validates the provider "
        "and the thinking level of a route, but NOT the model id, so a typo'd model only fails "
        "inside the spawned subagent."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "models", "search", "check", "refresh"],
                "description": ("list = every routable provider with model counts; models = the model "
                                "ids for one provider; search = fuzzy-match a fragment across all "
                                "providers; check = resolve one spec and show the exact route it "
                                "produces; refresh = rebuild the catalog."),
            },
            "provider": {"type": "string", "description": "Provider id, alias or label (for action=models)."},
            "query": {"type": "string", "description": "Model fragment (for action=search), e.g. 'ninfer', 'glm 5p3 flash'."},
            "route": {"type": "string", "description": "Spec to dry-run (for action=check), e.g. 'anthropic/claude-opus-5-5@max'."},
        },
        "required": ["action"],
    },
}


def _effort_set(provider: str, model: str) -> list[str]:
    """Levels the wire accepts for this pair (core clamps a stronger ask to the nearest weaker)."""
    try:
        from agent.reasoning_effort import route_supported_efforts
        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider(requested=provider, target_model=model)
        return list(route_supported_efforts(runtime.get("provider"), model,
                                            runtime.get("api_mode"), runtime.get("base_url")))
    except Exception:
        return []


def _entry(settings: Any, provider: str):
    return next((e for e in get_catalog(settings) if provider.lower() in (e.names() | {e.id.lower()})), None)


def make_handler(settings_provider: Callable[[], dict]) -> Callable[..., str]:
    """`settings_provider` returns the live `delegation.backends.dynamic` block per call."""

    def handler(args: dict, **kwargs) -> str:
        action = str(args.get("action") or "list").strip().lower()
        settings = settings_provider() or {}
        catalog = get_catalog(settings)

        if action == "refresh":
            catalog = refresh_catalog(settings)
            return json.dumps({"refreshed": True, "providers": len(catalog),
                               "models": sum(len(e.models) for e in catalog)})

        if action == "list":
            return json.dumps({"providers": [
                {"provider": e.id, "label": e.label, "kind": e.kind, "models": len(e.models),
                 "aliases": list(e.aliases)[:6]} for e in catalog]}, indent=1)

        if action == "models":
            name = str(args.get("provider") or "").strip()
            if not name:
                return json.dumps({"error": "action=models needs `provider`."})
            entry = _entry(settings, name)
            if entry is None:
                return json.dumps({"error": f"No routable provider matching {name!r}.",
                                   "providers": [e.id for e in catalog]})
            return json.dumps({"provider": entry.id, "kind": entry.kind, "count": len(entry.models),
                               "models": [{"id": m.id, "context_length": m.context_length,
                                           "vision": m.supports_vision} for m in entry.models]}, indent=1)

        if action == "search":
            query = str(args.get("query") or "").strip()
            if not query:
                return json.dumps({"error": "action=search needs `query`."})
            hits = []
            for entry in catalog:
                for model, score in cat._match_models(query, entry)[:5]:
                    hits.append({"score": score, "spec": f"{entry.id}/{model.id}",
                                 "provider": entry.id, "model": model.id,
                                 "context_length": model.context_length, "vision": model.supports_vision})
            hits.sort(key=lambda h: (-h["score"], len(h["model"])))
            return json.dumps({"query": query, "matches": hits[:20]}, indent=1)

        if action == "check":
            spec = str(args.get("route") or "").strip()
            if not spec:
                return json.dumps({"error": "action=check needs `route`."})
            try:
                resolved = cat.resolve(spec, catalog)
            except cat.RouteRefused as exc:
                return json.dumps({"ok": False, "spec": spec, "error": str(exc),
                                   "candidates": exc.candidates})
            efforts = _effort_set(resolved.provider, resolved.model)
            note = None
            if resolved.effort and efforts and resolved.effort not in efforts:
                weaker = [e for e in cat.EFFORTS if e in efforts]
                note = (f"{resolved.effort!r} is not offered by {resolved.model}; the wire clamps it "
                        f"to the nearest weaker offered level (offered: {', '.join(efforts)}).")
                resolved.effort = weaker[-1] if weaker else resolved.effort
            return json.dumps({"ok": True, "spec": spec, "canonical": resolved.canonical(),
                               "route": resolved.as_route(), "offered_efforts": efforts,
                               "context_length": resolved.context_length, "vision": resolved.supports_vision,
                               "note": note}, indent=1)

        return json.dumps({"error": f"Unknown action {action!r}. "
                                    "Use list, models, search, check or refresh."})

    return handler
