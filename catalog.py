"""Provider/model catalog + fuzzy route resolution.

Core refuses a delegation route whose *provider* has no credentials, and validates the
`reasoning_effort` level, but it never checks that the model id exists for that provider
(`validate_requested_model` is only wired into the /model switch path). So a typo'd model id
becomes a subagent that dies at its first API call. This module closes that gap: it builds the
list of providers the user actually has credentials for, with their models, and resolves the
agent's fuzzy spec (`og-home ninfer`) into a canonical `(provider, model)` pair — refusing with
ranked candidates when the guess is ambiguous.
"""

from __future__ import annotations

import difflib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
TIERS = ("normal", "fast", "ultrafast")
_TIER_WORDS = {"priority": "fast", "fast": "fast", "ultrafast": "ultrafast", "ultra": "ultrafast"}

#: A head token counts as a provider name only at exact/prefix/all-tokens strength, so a model
#: fragment that happens to contain a provider substring is not mis-split.
_PROVIDER_MIN = 60
#: Below this a model match is a guess, and a guess is refused with candidates rather than resolved:
#: core never validates model ids, so a silent near-match would spend a subagent on the wrong model.
_MODEL_MIN = 60

_ROUTE_RE = re.compile(r"^(?P<target>[^@+]*)(?:@(?P<effort>[A-Za-z]+))?(?P<plus>(?:\+[A-Za-z]+)*)$")


@dataclass
class ModelEntry:
    id: str
    context_length: Optional[int] = None
    supports_vision: Optional[bool] = None
    effort_set: tuple[str, ...] = ()
    preset_effort: Optional[str] = None
    preset_tier: Optional[str] = None

    def spec(self, provider: str) -> str:
        return f"{provider}/{self.id}"


@dataclass
class ProviderEntry:
    id: str
    label: str = ""
    kind: str = "catalog"           # "custom" (config.yaml providers:) | "catalog" (built-in)
    aliases: tuple[str, ...] = ()
    models: list[ModelEntry] = field(default_factory=list)
    base_url: str = ""
    #: True when `models` was narrowed to the Desktop picker's visible list, so a miss can say so.
    filtered: bool = False

    def names(self) -> set[str]:
        return {self.id.lower(), self.label.lower(), *(a.lower() for a in self.aliases)} - {""}


@dataclass
class Resolved:
    provider: str
    model: str
    effort: Optional[str] = None
    tier: Optional[str] = None
    daybreak: bool = False
    label: str = ""
    score: int = 0
    provider_kind: str = "catalog"
    context_length: Optional[int] = None
    supports_vision: Optional[bool] = None
    effort_set: tuple[str, ...] = ()

    def canonical(self) -> str:
        out = f"{self.provider}/{self.model}"
        if self.effort:
            out += f"@{self.effort}"
        if self.tier:
            out += f"+{self.tier}"
        if self.daybreak:
            out += "+daybreak"
        return out

    def as_route(self) -> dict[str, Any]:
        route: dict[str, Any] = {"provider": self.provider, "model": self.model,
                                 "label": self.label or self.canonical()}
        if self.effort:
            route["reasoning_effort"] = self.effort
        if self.tier:
            route["service_tier"] = self.tier
        if self.daybreak:
            route["daybreak"] = True
        return route


class RouteRefused(ValueError):
    """The spec cannot be resolved. The message is the tool error the agent reads, so it names fixes."""

    def __init__(self, message: str, candidates: Iterable[str] = ()):
        super().__init__(message)
        self.candidates = list(candidates)


# --- catalog construction ---------------------------------------------------------------------

def _config_custom_providers() -> list[ProviderEntry]:
    """providers:<name> in config.yaml — the authoritative model list for self-hosted endpoints."""
    try:
        from hermes_cli.config import load_config_readonly as load
    except ImportError:  # older cores
        from hermes_cli.config import load_config as load
    try:
        config = load() or {}
    except Exception:
        return []
    entries: list[ProviderEntry] = []
    providers = config.get("providers")
    if not isinstance(providers, dict):
        return entries
    for key, raw in providers.items():
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or key).strip()
        if not name:
            continue
        models: list[ModelEntry] = []
        declared = raw.get("models")
        if isinstance(declared, dict):
            for mid, meta in declared.items():
                meta = meta if isinstance(meta, dict) else {}
                models.append(ModelEntry(str(mid), meta.get("context_length"), meta.get("supports_vision")))
        elif isinstance(declared, list):
            models = [ModelEntry(str(m)) for m in declared]
        entries.append(ProviderEntry(id=name, label=str(raw.get("label") or name), kind="custom",
                                     aliases=tuple(a for a in (raw.get("aliases") or []) if isinstance(a, str)),
                                     base_url=str(raw.get("base_url") or raw.get("api") or "").strip(),
                                     models=models))
    return entries


def _read_models_cache() -> dict:
    import json
    import os
    path = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))) / "provider_models_cache.json"
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _cache_file_models(pid: str) -> list[str]:
    """Direct read of $HERMES_HOME/provider_models_cache.json, ignoring TTL. `cached_provider_model_ids`
    with non_blocking=True returns nothing once an entry is stale (it schedules a refresh instead),
    which would silently drop whole providers from the agent's view."""
    entry = _read_models_cache().get(pid)
    models = entry.get("models") if isinstance(entry, dict) else None
    return [str(m) for m in (models or []) if m]


def endpoint_catalog_models(base_url: str) -> list[ModelEntry]:
    """Live catalog Hermes fetched for a custom provider's ENDPOINT, cached under
    `custom:<base_url>#<credential-fingerprint>`. This is what the Desktop model picker shows, and a
    provider's hand-written `models:` list in config.yaml goes stale against it (pi-codex declared 6
    ids while the endpoint catalog listed 17, so `gpt-6-luna` was unresolvable and fuzzy-matched to
    `gpt-5.6-luna`)."""
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url:
        return []
    out: list[ModelEntry] = []
    for key, entry in _read_models_cache().items():
        if not key.startswith("custom:") or not isinstance(entry, dict):
            continue
        cached_url = key[len("custom:"):].split("#", 1)[0].rstrip("/")
        if cached_url.lower() != base_url.lower():
            continue
        for mid in entry.get("models") or []:
            if str(mid):
                out.append(ModelEntry(str(mid)))
    return out


VISIBLE_KEY = "hermes.desktop.visible-models"
KNOWN_KEY = "hermes.desktop.known-models"
PRESETS_KEY = "hermes.desktop.model-presets"


def _ui_values() -> dict:
    """`values` from the machine's UI-preferences store — the same file the Desktop app writes.
    Prefers core's lock-aware reader; falls back to a plain read for a Hermes without it."""
    import json
    try:
        from hermes_cli.ui_preferences import read_preferences
        data = read_preferences()
        values = data.get("values") if isinstance(data, dict) else None
        if isinstance(values, dict):
            return values
    except Exception:
        pass
    import os
    path = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))) / "ui-preferences.json"
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    values = data.get("values") if isinstance(data, dict) else None
    return values if isinstance(values, dict) else {}


def _decoded(values: dict, key: str, fallback):
    raw = values.get(key)
    if isinstance(raw, str):
        import json
        try:
            return json.loads(raw)
        except Exception:
            return fallback
    return raw if raw is not None else fallback


def _norm_provider_key(raw: str) -> str:
    """`anthropic`, `pi-codex`, and the legacy `custom:onlygens/<model>` form all name a provider;
    a bare `custom` names none."""
    raw = (raw or "").strip().lower()
    if not raw or raw == "custom":
        return ""
    if raw.startswith("custom:"):
        body = raw[len("custom:"):].split("#", 1)[0]
        return body.split("/", 1)[0].strip()
    return raw


def _pairs_from(values_key: str) -> dict[str, set[str]]:
    """Parse a `provider::model` list preference into `{provider: {model}}`."""
    items = _decoded(_ui_values(), values_key, [])
    out: dict[str, set[str]] = {}
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, str) or "::" not in item:
            continue
        prov, _, model = item.partition("::")
        prov, model = _norm_provider_key(prov), model.strip()
        if not prov or not model:
            continue
        out.setdefault(prov, set()).add(model)
    return out


def desktop_visible_models() -> dict[str, set[str]]:
    """The Desktop model picker's filter: `{provider_id: {model_id}}` from `visible-models`.
    Empty when the user has never filtered."""
    return _pairs_from(VISIBLE_KEY)


def desktop_known_models() -> dict[str, set[str]]:
    """Every `provider::model` key the user has had a chance to judge. The Desktop snapshots this
    each time the visible set is persisted, so a model absent from it appeared *after* the last
    curation (new release, plugin update) and falls through to the default-visible rule rather than
    defaulting to hidden. Mirrored here so the plugin and the picker agree on brand-new models."""
    return _pairs_from(KNOWN_KEY)


def desktop_model_presets() -> dict[tuple[str, str], dict]:
    """Per-model effort/tier the user set in the picker (`model-presets`)."""
    raw = _decoded(_ui_values(), PRESETS_KEY, {})
    out: dict[tuple[str, str], dict] = {}
    if not isinstance(raw, dict):
        return out
    for key, preset in raw.items():
        if not isinstance(key, str) or "::" not in key or not isinstance(preset, dict):
            continue
        prov, _, model = key.partition("::")
        prov, model = _norm_provider_key(prov), model.strip()
        if not prov or not model:
            continue
        out[(prov, model)] = dict(preset)
    return out


def _visible_for(entry: ProviderEntry, visible: dict[str, set[str]]) -> set[str]:
    for name in (entry.id.lower(), entry.label.lower(), *entry.aliases):
        if name in visible:
            return visible[name]
    return set()


def apply_desktop_filter(entries: list[ProviderEntry]) -> list[ProviderEntry]:
    """Narrow the catalog to the models the user shows in the Desktop picker, and attach their
    per-model effort/tier presets. Providers the picker lists but discovery missed are added."""
    visible = desktop_visible_models()
    if not visible:
        return entries
    known_all = desktop_known_models()
    presets = desktop_model_presets()
    seen: dict[str, ProviderEntry] = {e.id.lower(): e for e in entries}
    kept: list[ProviderEntry] = []
    for entry in entries:
        allowed = _visible_for(entry, visible)
        if not allowed:
            continue
        entry.filtered = True
        # Hidden = judged once (in known-models) and not visible. A model the user has never been
        # shown falls through to visible, exactly as the Desktop picker treats it.
        judged = _visible_for(entry, known_all)
        entry.models = [m for m in entry.models if m.id in allowed or m.id not in judged]
        for mid in sorted(allowed - {m.id for m in entry.models}):
            entry.models.append(ModelEntry(mid))
        for m in entry.models:
            preset = presets.get((entry.id.lower(), m.id))
            if preset:
                m.preset_effort = str(preset.get("effort") or "") or None
                tier = str(preset.get("serviceTier") or "")
                m.preset_tier = tier if tier and tier != "normal" else None
        if entry.models:
            kept.append(entry)
    for prov, models in visible.items():
        if prov in seen or not models:
            continue
        entry = ProviderEntry(id=prov, label=prov, kind="catalog", models=[], filtered=True)
        for mid in sorted(models):
            m = ModelEntry(mid)
            preset = presets.get((prov, mid))
            if preset:
                m.preset_effort = str(preset.get("effort") or "") or None
            entry.models.append(m)
        kept.append(entry)
        seen[prov] = entry
    return kept


def _catalog_provider_models(pid: str) -> list[ModelEntry]:
    from hermes_cli.models import cached_provider_model_ids, static_catalog
    ids: list[str] = []
    try:
        ids = list(cached_provider_model_ids(pid, non_blocking=True) or [])
    except Exception:
        ids = []
    if not ids:
        ids = _cache_file_models(pid)
    if not ids:
        try:
            ids = [str(m.get("id")) for m in (static_catalog(pid) or []) if isinstance(m, dict) and m.get("id")]
        except Exception:
            ids = []
    return [ModelEntry(mid) for mid in ids if mid]


def _cached_catalog_providers() -> list[str]:
    """Providers present in $HERMES_HOME/provider_models_cache.json — the ones whose model list was
    fetched with working credentials. This catches providers `_provider_has_credentials` misses,
    e.g. an Anthropic login borrowed from Claude Code."""
    import json
    import os
    path = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))) / "provider_models_cache.json"
    try:
        data = json.loads(path.read_text())
    except Exception:
        return []
    return [str(k) for k in data if isinstance(k, str) and not k.startswith("custom:")]


def provider_is_routable(name: str) -> bool:
    """Can a route naming this provider resolve credentials? This is core's own gate
    (`_resolve_delegation_credentials` -> `resolve_runtime_provider`), so the catalog stays advisory:
    the agent may name any provider it can actually authenticate, not only the ones we listed."""
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
    except Exception:
        return False
    try:
        resolve_runtime_provider(requested=name, target_model="probe")
        return True
    except Exception:
        return False


def build_catalog(include: Optional[set[str]] = None, exclude: Optional[set[str]] = None,
                  extra_providers: Iterable[str] = (), respect_desktop_filter: bool = True) -> list[ProviderEntry]:
    """Every provider the user can likely route to: configured custom endpoints, authenticated
    catalog providers, and providers with a cached (credential-fingerprinted) model list.
    Never touches the network (cache + static catalog only)."""
    from hermes_cli.models import list_available_providers

    entries: dict[str, ProviderEntry] = {}
    for pe in _config_custom_providers():
        entries[pe.id.lower()] = pe
    try:
        available = list_available_providers()
    except Exception:
        available = []
    by_id = {str(r.get("id")): r for r in available}
    candidates = [row for row in available if row.get("authenticated")]
    candidates += [{"id": pid, "label": pid, "aliases": []} for pid in _cached_catalog_providers()
                   if pid in by_id]
    for row in candidates + [{"id": str(x), "label": str(x), "aliases": []} for x in extra_providers]:
        pid = str(row.get("id") or "")
        if not pid or pid == "custom" or pid.lower() in entries:
            continue
        models = _catalog_provider_models(pid)
        if not models:
            continue
        aliases = tuple(row.get("aliases") or ())
        prev = entries.get(pid.lower())
        entries[pid.lower()] = ProviderEntry(
            id=pid, label=str(row.get("label") or pid), kind="catalog",
            aliases=tuple(dict.fromkeys((prev.aliases if prev else ()) + aliases)), models=models)

    # A custom provider's hand-written `models:` list goes stale against the live catalog Hermes
    # fetched for that endpoint (what the Desktop picker shows). Merge both, keeping the richer
    # config entry (context length, vision) when an id appears in each.
    for entry in entries.values():
        if entry.kind != "custom" or not entry.base_url:
            continue
        by_id = {m.id.lower(): m for m in entry.models}
        for extra in endpoint_catalog_models(entry.base_url):
            known = by_id.get(extra.id.lower())
            if known is not None:
                known.context_length = known.context_length or extra.context_length
                known.supports_vision = known.supports_vision if known.supports_vision is not None else extra.supports_vision
                continue
            entry.models.append(extra)
            by_id[extra.id.lower()] = extra

    out = list(entries.values())
    if include:
        keep = {s.lower() for s in include}
        out = [e for e in out if keep & (e.names() | {e.id.lower()})]
    if exclude:
        drop = {s.lower() for s in exclude}
        out = [e for e in out if not (drop & (e.names() | {e.id.lower()}))]
    if respect_desktop_filter:
        out = apply_desktop_filter(out)
    return sorted(out, key=lambda e: (e.kind != "custom", e.id))


# --- matching ---------------------------------------------------------------------------------

def _score(needle: str, hay: str) -> int:
    """0 = no match. Higher is better. Every level above the refusal floor is a *token-boundary*
    match, so a short id is never absorbed by a longer one: `kimi-k3` must not resolve to
    `kimi-k3p1-pro` (substring matching allowed exactly that)."""
    if not needle or not hay:
        return 0
    n, h = needle.lower().strip(), hay.lower()
    if n == h:
        return 100
    ntok = [t for t in re.split(r"[^a-z0-9]+", n) if t]
    htok = [t for t in re.split(r"[^a-z0-9]+", h) if t]
    if h.startswith(n) and (len(h) == len(n) or not h[len(n)].isalnum()):
        return 80                       # prefix ending on a token boundary: "glm-5p3" -> "glm-5p3-flash"
    if ntok and htok and all(t in htok for t in ntok):
        return 70                       # every token of the hint is a whole token of the id
    # Word-prefix on every token: "kimi k3 pro" -> "kimi-k3p1-pro". Lenient by design — the user
    # addresses models by words, and an exact id always outscores it (100), so a real `kimi-k3`
    # present in the list wins over this.
    if ntok and htok and all(any(t == u or (len(t) >= 2 and u.startswith(t)) for u in htok) for t in ntok):
        return 65
    if n in h:
        return 40                       # substring: below the floor, a candidate, never a resolve
    ratio = difflib.SequenceMatcher(None, n, h).ratio()
    return int(ratio * 30) if ratio >= 0.6 else 0


def _match_provider(hint: str, catalog: list[ProviderEntry]) -> tuple[Optional[ProviderEntry], int]:
    best, score = None, 0
    for entry in catalog:
        candidates = [_score(hint, entry.id), _score(hint, entry.label),
                      *(_score(hint, a) for a in entry.aliases)]
        s = max(candidates)
        if s > score:
            best, score = entry, s
    return (best, score) if score >= _PROVIDER_MIN else (None, 0)


def _match_models(hint: str, entry: ProviderEntry) -> list[tuple[ModelEntry, int]]:
    scored = [(m, _score(hint, m.id)) for m in entry.models]
    hits = [(m, s) for m, s in scored if s > 0]
    hits.sort(key=lambda pair: (-pair[1], len(pair[0].id)))
    return hits


def _split_target(target: str, catalog: list[ProviderEntry]) -> tuple[Optional[ProviderEntry], str]:
    """Split a spec into (provider, model-hint). The provider may be multi-word ("Fireworks AI"), so
    every prefix is tried and the longest confident provider match wins. No provider match means the
    whole string is a model hint, searched across all providers."""
    target = target.strip()
    best: tuple[int, int, Optional[ProviderEntry], str] = (0, 0, None, target)
    for sep in ("/", " ", ":"):
        if sep not in target:
            continue
        parts = target.split(sep)
        for i in range(1, min(len(parts), 5)):
            head = sep.join(parts[:i]).strip()
            tail = sep.join(parts[i:]).strip()
            if not head or not tail:
                continue
            entry, score = _match_provider(head, catalog)
            # On equal strength the longer head wins: "Fireworks AI" beats "Fireworks".
            if entry is not None and (score, len(head)) > (best[0], best[1]):
                best = (score, len(head), entry, tail)
            # A provider named last ("luna 6 pi-codex") is as valid as one named first.
            head2 = sep.join(parts[-i:]).strip()
            tail2 = sep.join(parts[:-i]).strip()
            if head2 and tail2:
                entry2, score2 = _match_provider(head2, catalog)
                if entry2 is not None and (score2, len(head2)) > (best[0], best[1]):
                    best = (score2, len(head2), entry2, tail2)
    if best[0] >= _PROVIDER_MIN:
        return best[2], best[3]
    for sep in ("/", " ", ":"):
        if sep in target:
            head, _, tail = target.partition(sep)
            if head.strip() and tail.strip():
                entry, _ = _match_provider(head.strip(), catalog)
                if entry is None:
                    return None, target.strip()
    return None, target.strip()


def parse_spec(spec: str) -> tuple[str, Optional[str], Optional[str], bool]:
    """`provider/model@effort+tier+daybreak` -> (target, effort, tier, daybreak)."""
    m = _ROUTE_RE.match(spec.strip())
    if not m:
        raise RouteRefused(f"Cannot parse route {spec!r}. Use provider/model@effort+tier, "
                           f"e.g. anthropic/claude-opus-5-5@max.")
    target = (m.group("target") or "").strip()
    effort = (m.group("effort") or "").strip().lower() or None
    if effort and effort not in EFFORTS:
        raise RouteRefused(f"Unknown thinking level {effort!r} in {spec!r}. "
                           f"Use one of: {', '.join(EFFORTS)}.")
    tier: Optional[str] = None
    daybreak = False
    for token in [t.lstrip("+").strip().lower() for t in (m.group("plus") or "").split("+") if t.strip()]:
        if token == "daybreak":
            daybreak = True
        elif token in _TIER_WORDS:
            tier = _TIER_WORDS[token]
        else:
            raise RouteRefused(f"Unknown modifier {token!r} in {spec!r}. "
                               f"Supported: +fast, +ultrafast, +daybreak.")
    if not target:
        raise RouteRefused(f"Route {spec!r} has no provider/model part.")
    return target, effort, tier, daybreak


def resolve(spec: str, catalog: list[ProviderEntry], parent: Optional[tuple[str, str]] = None,
            aliases: Optional[dict[str, str]] = None) -> Resolved:
    """The agent's free-form spec -> a canonical route. Refuses with candidates when ambiguous."""
    spec = (spec or "").strip()
    if not spec:
        raise RouteRefused("Empty route.")
    if aliases and spec.lower() in {k.lower(): v for k, v in aliases.items()}:
        spec = {k.lower(): v for k, v in aliases.items()}[spec.lower()]

    target, effort, tier, daybreak = parse_spec(spec)
    provider_hint, model_hint = _split_target(target, catalog)

    if provider_hint is None:
        # The agent may name a provider our discovery missed (a borrowed login, a fresh key). Core's
        # own resolver is the authority, so try the head as a provider before refusing. Model ids
        # cannot be validated for such a provider — it is reported as "unverified".
        # When the Desktop picker's filter is in force it is the authority instead: a provider the
        # user does not show there is not routable, so no fallback.
        if not any(e.filtered for e in catalog):
            for sep in ("/", " ", ":"):
                if sep not in target:
                    continue
                head, _, tail = target.partition(sep)
                head, tail = head.strip(), tail.strip()
                if head and tail and provider_is_routable(head):
                    return Resolved(provider=head, model=tail, effort=effort, tier=tier, daybreak=daybreak,
                                    label=spec, provider_kind="unverified")

    if provider_hint is None and parent and parent[0]:
        entry, score = _match_provider(parent[0], catalog)
        provider_hint = entry if entry and _score(model_hint, entry.id) == 0 else None
        if provider_hint is None:
            provider_hint = next((e for e in catalog if e.id.lower() == parent[0].lower()), None)

    if provider_hint is not None:
        hits = _match_models(model_hint, provider_hint)
        if not hits:
            near = difflib.get_close_matches(model_hint, [m.id for m in provider_hint.models], n=5, cutoff=0.4)
            scope = ("your Desktop model filter" if provider_hint.filtered
                     else f"{len(provider_hint.models)} models configured")
            raise RouteRefused(
                f"No model matching {model_hint!r} on provider {provider_hint.id!r} ({scope}). "
                f"Closest: {', '.join(near) or 'none'}. "
                f"Call dynamic_delegation_models(action='models', provider={provider_hint.id!r}) for the list.",
                near)
        best_model, mscore = hits[0]
        tied = [m for m, s in hits if s == mscore]
        if mscore < _MODEL_MIN or len(tied) > 1:
            scope = " (only models you show in the Desktop picker are routable)" if provider_hint.filtered else ""
            raise RouteRefused(
                f"{model_hint!r} is not a confident match on {provider_hint.id}{scope}. "
                f"Candidates: " + ", ".join(f"{provider_hint.id}/{m.id}" for m in tied[:6]),
                [f"{provider_hint.id}/{m.id}" for m in tied[:6]])
        resolved_provider, resolved_model = provider_hint, best_model
    else:
        pooled: list[tuple[ProviderEntry, ModelEntry, int]] = []
        for entry in catalog:
            for model, score in _match_models(model_hint, entry):
                pooled.append((entry, model, score))
        if not pooled:
            raise RouteRefused(
                f"No model matching {model_hint!r} on any provider you have credentials for "
                f"({len(catalog)} providers searched). Call dynamic_delegation_models(action='search', "
                f"query={model_hint!r}) to see candidates, or name the provider: provider/model.")
        pooled.sort(key=lambda t: (-t[2], len(t[1].id)))
        top = pooled[0][2]
        tied = [t for t in pooled if t[2] == top]
        if top < _MODEL_MIN or len({(e.id, m.id) for e, m, _ in tied}) > 1:
            raise RouteRefused(f"{model_hint!r} matches several providers/models. Name one: "
                               + ", ".join(f"{e.id}/{m.id}" for e, m, _ in tied[:6]),
                               [f"{e.id}/{m.id}" for e, m, _ in tied[:6]])
        resolved_provider, resolved_model, _ = tied[0]

    # No explicit @effort => the level the user set for that model in the Desktop picker, when they set one.
    if not effort and resolved_model.preset_effort:
        effort = resolved_model.preset_effort
    if not tier and resolved_model.preset_tier:
        tier = resolved_model.preset_tier

    out = Resolved(provider=resolved_provider.id, model=resolved_model.id, effort=effort, tier=tier,
                   daybreak=daybreak, label=spec, provider_kind=resolved_provider.kind,
                   context_length=resolved_model.context_length, supports_vision=resolved_model.supports_vision,
                   effort_set=resolved_model.effort_set)
    if daybreak and out.provider != "openai-codex":
        raise RouteRefused(f"+daybreak needs provider openai-codex, not {out.provider!r}.")
    return out
