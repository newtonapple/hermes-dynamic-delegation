"""The `dynamic` delegation backend.

`delegate_task` task items gain one free-form string parameter `route`, which the agent fills with
`provider/model@effort+tier`. Core enforces only `enum` on provider-declared parameters
(tools/delegate_tool_routes.py:_param_error), and `provider`/`model`/`base_url`/`api_key`/`api_mode`
are reserved names a provider may not declare (agent/delegation_provider.py:RESERVED_TASK_PARAMETERS)
— so the whole selection travels as one string this provider decodes into a route dict.

Core still validates what we hand back: an unknown provider is refused per task (credential
resolution), an unknown `reasoning_effort` is refused, and a `service_tier` the route cannot serve
is refused. Core does NOT validate model ids, so `catalog.resolve` does that here and refuses with
ranked candidates instead of letting the child die at its first API call.

`base_url` and `api_key` are never parsed from agent input: the grammar has no slot for them, so an
agent can only reach providers you have already authenticated.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from agent.delegation_provider import DelegationProvider, DelegationRouteError

from . import catalog as cat

logger = logging.getLogger(__name__)

BACKEND = "dynamic"
#: Named routes from the official advanced-delegation model_map block, usable as shortcuts.
SHORTCUT_BLOCK = "model_map"
#: Route keys core accepts (agent/delegation_provider.py:ROUTE_KEYS), minus the ones an agent must
#: never supply through a shortcut: profile/label are set here, api_key stays out of plugin config.
ROUTE_FIELDS = ("provider", "model", "base_url", "api_mode", "reasoning_effort", "service_tier")
_cache: dict[str, Any] = {"ts": 0.0, "catalog": []}


def _settings(config: Mapping[str, Any]) -> dict:
    backends = config.get("backends") if isinstance(config, Mapping) else None
    block = backends.get(BACKEND) if isinstance(backends, Mapping) else None
    return dict(block) if isinstance(block, Mapping) else {}


def get_catalog(settings: Mapping[str, Any], force: bool = False) -> list[cat.ProviderEntry]:
    """Cached provider/model catalog (never blocks on the network). Keyed on the settings that
    change its shape, so a `delegation config` edit takes effect without a restart."""
    ttl = float(settings.get("catalog_ttl_seconds", 300) or 300)
    key = repr(
        (
            tuple(sorted(str(x) for x in (settings.get("allow_providers") or []))),
            tuple(sorted(str(x) for x in (settings.get("deny_providers") or []))),
            tuple(sorted(str(x) for x in (settings.get("extra_providers") or []))),
            bool(settings.get("respect_desktop_filter", True)),
        )
    )
    now = time.time()
    if not force and _cache["catalog"] and _cache.get("key") == key and (now - _cache["ts"]) < ttl:
        return _cache["catalog"]
    include = settings.get("allow_providers") or None
    exclude = settings.get("deny_providers") or None
    entries = cat.build_catalog(
        include={str(x) for x in include} if include else None,
        exclude={str(x) for x in exclude} if exclude else None,
        extra_providers=[str(x) for x in (settings.get("extra_providers") or [])],
        respect_desktop_filter=bool(settings.get("respect_desktop_filter", True)),
    )
    _cache.update(ts=now, catalog=entries, key=key)
    return entries


def _daybreak_route(name: str, route: dict[str, Any]) -> dict[str, Any]:
    """Reuse the official plugin's Daybreak gate (refuses rather than silently dropping the program)."""
    try:
        home = Path.home() / ".hermes" / "plugins" / "advanced-delegation" / "model_map.py"
        spec = importlib.util.spec_from_file_location("_adv_deleg_model_map", home)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {home}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["_adv_deleg_model_map"] = module
        spec.loader.exec_module(module)
        return module._daybreak_route(name, route)
    except Exception as exc:
        raise DelegationRouteError(
            f"Route {name!r} asks for Daybreak, which needs the advanced-delegation plugin's gate "
            f"({exc}). Install advanced-delegation or drop +daybreak."
        ) from exc


class DynamicDelegationProvider(DelegationProvider):
    """`delegation.backend: dynamic` — the agent names provider, model, thinking level and tier per task."""

    @property
    def name(self) -> str:
        return BACKEND

    @property
    def display_name(self) -> str:
        return "Dynamic model routes"

    @property
    def summary(self) -> str:
        return (
            "The agent writes provider/model@effort+tier per subagent, resolved against every "
            "provider you have credentials for."
        )

    # --- schema ---------------------------------------------------------------------------

    def task_parameters(self, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        settings = self.settings(config)
        entries = get_catalog(settings)
        providers = ", ".join(e.id for e in entries) or "(none detected)"
        lines = [
            "Which model runs this subagent, as 'provider/model@effort+tier'.",
            "Provider may be an id, alias or label; the model may be a fragment "
            "(e.g. 'og-home ninfer' -> og-home/qwen3.8-27b-ninfer).",
            "Effort: "
            + ", ".join(cat.EFFORTS)
            + ". Tiers: +fast, +ultrafast, +daybreak (openai-codex only).",
            "Omit to inherit the delegating agent's own model. A level the model does not offer is "
            "clamped to the nearest weaker one, never stronger.",
            "Call dynamic_delegation_models to list, search or dry-run a spec before you delegate.",
            "Providers available: " + providers,
        ]
        return {"route": {"type": "string", "description": "\n".join(lines)}}

    def description(self, config: Mapping[str, Any]) -> str:  # noqa: ARG002 - core's signature
        return (
            "Each task may set `route` to run that subagent on an exact provider/model at an "
            'exact thinking level, e.g. route: "anthropic/claude-opus-5@max".'
        )

    # --- routing --------------------------------------------------------------------------

    def _shortcut(self, spec: str, config: Mapping[str, Any]) -> Optional[dict[str, Any]]:
        """A bare name from the official model_map options list, so named routes keep working."""
        backends = config.get("backends") if isinstance(config, Mapping) else {}
        block = backends.get(SHORTCUT_BLOCK) if isinstance(backends, Mapping) else {}
        options = block.get("options") if isinstance(block, Mapping) else None
        if not isinstance(options, list):
            return None
        wanted = spec.strip().lower()
        for entry in options:
            if not isinstance(entry, Mapping) or str(entry.get("name", "")).strip().lower() != wanted:
                continue
            route: dict[str, Any] = {
                key: str(entry[key]).strip()
                for key in ROUTE_FIELDS
                if isinstance(entry.get(key), str) and entry[key].strip()
            }
            route["label"] = spec.strip()
            if entry.get("daybreak") is True:
                route["daybreak"] = True
            return route
        return None

    def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        settings = self.settings(config)
        raw = task.get("route")
        if not isinstance(raw, str) or not raw.strip():
            default = str(settings.get("default") or "").strip()
            if not default:
                return {}  # inherit the parent's route
            raw = default
        spec = raw.strip()

        shortcut = self._shortcut(spec, config)
        if shortcut is not None:
            return shortcut

        parent = self._parent_route(config)
        try:
            resolved = cat.resolve(
                spec, get_catalog(settings), parent=parent, aliases=settings.get("aliases") or {}
            )
        except cat.RouteRefused as exc:
            raise DelegationRouteError(str(exc)) from exc

        route = resolved.as_route()
        if resolved.daybreak:
            route = _daybreak_route(resolved.canonical(), route)
        return route

    @staticmethod
    def _parent_route(config: Mapping[str, Any]) -> Optional[tuple[str, str]]:  # noqa: ARG004
        """(provider, model) of the delegating agent, used when a spec names only a model. Core hands
        a delegation provider the `delegation` section, so the agent's own model comes from config."""
        try:
            from hermes_cli.config import load_config_readonly as load
        except ImportError:
            from hermes_cli.config import load_config as load
        try:
            model_cfg = (load() or {}).get("model") or {}
        except Exception:
            return None
        provider = str(model_cfg.get("provider") or "").strip()
        base_url = str(model_cfg.get("base_url") or "").strip()
        if not provider and not base_url:
            return None
        return (provider or "custom", str(model_cfg.get("default") or "").strip())


def refresh_catalog(settings: Optional[Mapping[str, Any]] = None) -> list[cat.ProviderEntry]:
    return get_catalog(settings or {}, force=True)
