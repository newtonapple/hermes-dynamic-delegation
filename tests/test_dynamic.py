"""Tests for dynamic-delegation. Run from the repo root:

python -m pytest -q            # needs Hermes core importable for the integration cases
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

PKG_DIR = Path(__file__).resolve().parents[1]


def _load_package() -> object:
    spec = importlib.util.spec_from_file_location(
        "dynamic_delegation", PKG_DIR / "__init__.py", submodule_search_locations=[str(PKG_DIR)]
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["dynamic_delegation"] = module
    spec.loader.exec_module(module)
    for sub in ("catalog", "provider", "discovery"):
        setattr(module, sub, importlib.import_module(f"dynamic_delegation.{sub}"))
    return module


try:
    sys.path.insert(0, str(Path.home() / ".hermes" / "hermes-agent"))
    import agent.delegation_provider  # noqa: F401

    CORE = True
except ImportError:
    CORE = False

_pkg = _load_package()
cat = _pkg.catalog


def _catalog() -> list:
    return [
        cat.ProviderEntry(
            id="og-home",
            label="og-home",
            kind="custom",
            models=[
                cat.ModelEntry("qwen3.8-27b-ninfer", 262144, True),
                cat.ModelEntry("qwen3.8-flash-halogen", 1048576, True),
                cat.ModelEntry("glm5.3-flash", 1048576, True),
            ],
        ),
        cat.ProviderEntry(
            id="onlygens",
            label="onlygens",
            kind="custom",
            models=[
                cat.ModelEntry("qwen3.8-27b-ninfer", 262144, True),
                cat.ModelEntry("qwen3.8-flash-halogen", 1048576, True),
            ],
        ),
        cat.ProviderEntry(
            id="anthropic",
            label="Anthropic",
            kind="catalog",
            aliases=("claude", "claude-code"),
            models=[
                cat.ModelEntry("claude-opus-5-5", 1048576, True),
                cat.ModelEntry("claude-sonnet-5-5", 1048576, True),
                cat.ModelEntry("claude-haiku-5-5", 1048576, True),
            ],
        ),
        cat.ProviderEntry(
            id="fireworks",
            label="Fireworks AI",
            kind="catalog",
            aliases=("fireworks-ai", "fw"),
            models=[
                cat.ModelEntry("accounts/fireworks/models/glm-5p3-flash"),
                cat.ModelEntry("accounts/fireworks/routers/glm-5p3-fast"),
            ],
        ),
    ]


# --- grammar ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec,target,effort,tier,db",
    [
        ("anthropic/claude-opus-5-5@max", "anthropic/claude-opus-5-5", "max", None, False),
        ("fireworks glm-5p3-flash@high+fast", "fireworks glm-5p3-flash", "high", "fast", False),
        (
            "openai-codex/gpt-6-sol@high+ultrafast+daybreak",
            "openai-codex/gpt-6-sol",
            "high",
            "ultrafast",
            True,
        ),
        ("og-home ninfer", "og-home ninfer", None, None, False),
        ("claude-opus-5-5", "claude-opus-5-5", None, None, False),
    ],
)
def test_parse_spec(spec, target, effort, tier, db):
    assert cat.parse_spec(spec) == (target, effort, tier, db)


def test_parse_spec_refuses_bad_level_and_modifier():
    with pytest.raises(cat.RouteRefused):
        cat.parse_spec("anthropic/claude-opus-5-5@ultraa")
    with pytest.raises(cat.RouteRefused):
        cat.parse_spec("anthropic/claude-opus-5-5@high+turbo")


# --- resolution -------------------------------------------------------------------------------


def test_fuzzy_provider_and_model_fragment():
    r = cat.resolve("og-home ninfer", _catalog())
    assert (r.provider, r.model) == ("og-home", "qwen3.8-27b-ninfer")


def test_canonical_form_with_effort_and_tier():
    r = cat.resolve("fireworks/glm-5p3-flash@high+fast", _catalog())
    assert r.provider == "fireworks"
    assert r.model == "accounts/fireworks/models/glm-5p3-flash"
    assert r.canonical() == "fireworks/accounts/fireworks/models/glm-5p3-flash@high+fast"
    assert r.as_route()["reasoning_effort"] == "high"
    assert r.as_route()["service_tier"] == "fast"


def test_provider_alias_and_label():
    assert cat.resolve("claude opus 5.5", _catalog()).provider == "anthropic"
    assert cat.resolve("Fireworks AI glm-5p3-flash", _catalog()).provider == "fireworks"


def test_bare_model_unique_across_providers():
    assert cat.resolve("opus-5-5@max", _catalog()).provider == "anthropic"


def test_bare_model_ambiguous_names_the_contenders():
    with pytest.raises(cat.RouteRefused) as exc:
        cat.resolve("ninfer", _catalog())
    assert "og-home/qwen3.8-27b-ninfer" in str(exc.value)
    assert set(exc.value.candidates) == {"og-home/qwen3.8-27b-ninfer", "onlygens/qwen3.8-27b-ninfer"}


def test_unknown_model_refuses_with_closest():
    with pytest.raises(cat.RouteRefused) as exc:
        cat.resolve("anthropic/claude-opus-9", _catalog())
    assert "claude-opus-5-5" in str(exc.value)


def test_unknown_provider_name_is_not_a_provider():
    # 'bogus' is not a provider, so the whole string is treated as a model search.
    with pytest.raises(cat.RouteRefused):
        cat.resolve("bogus/nope@high", _catalog())


def test_daybreak_requires_openai_codex():
    with pytest.raises(cat.RouteRefused):
        cat.resolve("anthropic/claude-opus-5-5@high+daybreak", _catalog())


def test_alias_map_shortcut():
    r = cat.resolve("deep", _catalog(), aliases={"deep": "anthropic/claude-opus-5-5@max"})
    assert (r.provider, r.model, r.effort) == ("anthropic", "claude-opus-5-5", "max")


def test_empty_spec_refuses():
    with pytest.raises(cat.RouteRefused):
        cat.resolve("   ", _catalog())


def test_endpoint_catalog_models_reads_the_custom_key(tmp_path, monkeypatch):
    """The Desktop picker's list for a custom provider lives under `custom:<base_url>#<fingerprint>`."""
    import json

    (tmp_path / "provider_models_cache.json").write_text(
        json.dumps(
            {
                "custom:https://chatgpt.com/backend-api/codex#b31060bd50519d8a": {
                    "fp": "b31060bd50519d8a",
                    "models": ["gpt-6.2-sol", "gpt-6-luna", "gpt-6-luna-900k"],
                },
                "fireworks": {"models": ["accounts/fireworks/models/kimi-k3"]},
            }
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ids = [m.id for m in cat.endpoint_catalog_models("https://chatgpt.com/backend-api/codex/")]
    assert ids == ["gpt-6.2-sol", "gpt-6-luna", "gpt-6-luna-900k"]
    assert cat.endpoint_catalog_models("https://nope.example/v1") == []
    assert cat.endpoint_catalog_models("") == []


def test_exact_id_wins_over_a_fuzzy_neighbour():
    """`gpt-6-luna` resolves exactly; the fuzzy form `luna 6` is genuinely ambiguous (it token-matches
    gpt-5.6-luna too) and must refuse with candidates rather than pick a generation for the agent."""
    entry = cat.ProviderEntry(
        id="pi-codex",
        label="pi-codex",
        kind="custom",
        models=[cat.ModelEntry("gpt-5.6-luna"), cat.ModelEntry("gpt-6-luna"), cat.ModelEntry("gpt-6.2-sol")],
    )
    assert cat.resolve("pi-codex/gpt-6-luna", [entry]).model == "gpt-6-luna"
    with pytest.raises(cat.RouteRefused) as exc:
        cat.resolve("pi-codex luna 6", [entry])
    assert {"pi-codex/gpt-5.6-luna", "pi-codex/gpt-6-luna"} <= set(exc.value.candidates)


def test_fuzzy_words_resolve_and_an_exact_id_always_wins():
    """Word matching is the point: `kimi k3 pro` reaches kimi-k3p1-pro, and when a real `kimi-k3`
    is in the list the exact id outscores the fuzzy neighbour."""
    entry = cat.ProviderEntry(
        id="fireworks",
        label="Fireworks AI",
        kind="catalog",
        models=[
            cat.ModelEntry("accounts/fireworks/models/kimi-k3p1-pro"),
            cat.ModelEntry("accounts/fireworks/models/kimi-k3"),
            cat.ModelEntry("accounts/fireworks/models/glm-5p3-flash"),
        ],
    )
    assert cat.resolve("fireworks/kimi-k3", [entry]).model == "accounts/fireworks/models/kimi-k3"
    assert cat.resolve("fireworks kimi k3 pro", [entry]).model.endswith("kimi-k3p1-pro")
    assert cat.resolve("fireworks/glm 5p3 flash", [entry]).model == "accounts/fireworks/models/glm-5p3-flash"


def test_provider_named_last_is_detected():
    entry = cat.ProviderEntry(
        id="pi-codex",
        label="pi-codex",
        kind="custom",
        models=[cat.ModelEntry("gpt-6-luna"), cat.ModelEntry("gpt-5.6-luna")],
    )
    for spec in ["pi-codex gpt-6-luna", "gpt-6-luna pi-codex", "pi-codex: gpt-6-luna"]:
        assert cat.resolve(spec, [entry]).model == "gpt-6-luna", spec
    # The word form stays deliberately ambiguous: `luna 6` token-matches 6-luna and 5.6-luna alike,
    # so it refuses with both candidates rather than picking a generation.
    with pytest.raises(cat.RouteRefused):
        cat.resolve("pi-codex luna 6", [entry])


# --- the Desktop picker's filter ---------------------------------------------------------------


def _write_ui_prefs(tmp_path, visible, presets=None, known=None) -> None:
    import json

    values = {"hermes.desktop.visible-models": json.dumps(visible)}
    if presets is not None:
        values["hermes.desktop.model-presets"] = json.dumps(presets)
    if known is not None:
        values["hermes.desktop.known-models"] = json.dumps(known)
    (tmp_path / "ui-preferences.json").write_text(json.dumps({"values": values}))


def test_models_never_judged_fall_through_to_visible(tmp_path, monkeypatch):
    """The Desktop rule: hidden = judged once and not shown. A model that arrived after the last
    curation is absent from known-models, so it stays routable rather than defaulting to hidden."""
    _write_ui_prefs(
        tmp_path,
        visible=["anthropic::claude-haiku-5-5"],
        known=["anthropic::claude-haiku-5-5", "anthropic::claude-opus-5-5", "anthropic::claude-sonnet-5-5"],
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entries = cat.apply_desktop_filter(_catalog())
    anthropic = next(e for e in entries if e.id == "anthropic")
    ids = {m.id for m in anthropic.models}
    assert "claude-haiku-5-5" in ids  # visible
    assert "claude-opus-5-5" not in ids  # judged and hidden
    assert "claude-sonnet-5-5" not in ids  # judged and hidden
    # A provider entry carrying a model the user has never seen keeps it.
    extra = cat.ProviderEntry(
        id="anthropic",
        label="Anthropic",
        kind="catalog",
        models=[cat.ModelEntry("claude-haiku-5-5"), cat.ModelEntry("claude-brand-new-9")],
    )
    kept = next(e for e in cat.apply_desktop_filter([extra]) if e.id == "anthropic")
    assert "claude-brand-new-9" in {m.id for m in kept.models}


def test_filter_narrows_models_and_keeps_preset_effort(tmp_path, monkeypatch):
    _write_ui_prefs(
        tmp_path,
        ["anthropic::claude-opus-5-5", "anthropic::claude-haiku-5-5", "pi-codex::gpt-6-luna"],
        {"anthropic::claude-opus-5-5": {"effort": "high", "serviceTier": "normal"}},
        known=[
            "anthropic::claude-opus-5-5",
            "anthropic::claude-haiku-5-5",
            "anthropic::claude-sonnet-5-5",
            "pi-codex::gpt-6-luna",
            "pi-codex::gpt-5.6-luna",
        ],
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entries = cat.apply_desktop_filter(
        _catalog()
        + [
            cat.ProviderEntry(
                id="pi-codex", label="pi-codex", kind="custom", models=[cat.ModelEntry("gpt-5.6-luna")]
            )
        ]
    )
    by_id = {e.id: e for e in entries}
    assert sorted(m.id for m in by_id["anthropic"].models) == ["claude-haiku-5-5", "claude-opus-5-5"]
    assert by_id["anthropic"].filtered is True
    # A provider the picker lists but discovery missed is added, so the filter is the authority.
    assert [m.id for m in by_id["pi-codex"].models] == ["gpt-6-luna"]
    # Providers with nothing visible are dropped entirely.
    assert "og-home" not in by_id and "fireworks" not in by_id
    assert cat.resolve("anthropic/claude-opus-5-5", entries).effort == "high"
    assert cat.resolve("anthropic/claude-opus-5-5@max", entries).effort == "max"


def test_filter_refuses_a_model_the_user_hid(tmp_path, monkeypatch):
    _write_ui_prefs(
        tmp_path,
        ["anthropic::claude-haiku-5-5"],
        known=["anthropic::claude-haiku-5-5", "anthropic::claude-opus-5-5"],
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entries = cat.apply_desktop_filter(_catalog())
    with pytest.raises(cat.RouteRefused) as exc:
        cat.resolve("anthropic/claude-opus-5-5", entries)
    assert "Desktop" in str(exc.value)
    assert cat.resolve("anthropic/claude-haiku-5-5", entries).model == "claude-haiku-5-5"


def test_filter_blocks_the_routable_provider_fallback(tmp_path, monkeypatch):
    """With a filter in force, a provider the user does not show is not routable, even with credentials."""
    _write_ui_prefs(tmp_path, ["anthropic::claude-haiku-5-5"])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entries = cat.apply_desktop_filter(_catalog())
    assert any(e.filtered for e in entries)
    with pytest.raises(cat.RouteRefused):
        cat.resolve("some-provider-i-can-authenticate/x", entries)


def test_no_filter_file_leaves_the_catalog_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    entries = cat.apply_desktop_filter(_catalog())
    assert {e.id for e in entries} == {"og-home", "onlygens", "anthropic", "fireworks"}
    assert not any(e.filtered for e in entries)


# --- the provider, on core's real code paths ---------------------------------------------------


@pytest.mark.skipif(not CORE, reason="needs Hermes core importable")
def test_core_accepts_the_free_form_parameter_and_the_route_it_returns():
    from tools.delegate_tool_routes import (
        _check_backend_effort,
        _clean_route,
        _declared_parameters,
        _param_error,
    )

    DynamicDelegationProvider = _pkg.provider.DynamicDelegationProvider

    provider = DynamicDelegationProvider()
    # Hermetic: never read the user's live Desktop picker state from a unit test.
    config = {"backend": "dynamic", "backends": {"dynamic": {"respect_desktop_filter": False}}}
    declared = _declared_parameters(provider, config)
    assert "route" in declared, declared
    assert "model" not in declared and "provider" not in declared
    # No enum => core performs no value check on the agent's string.
    assert _param_error(0, "route", "anthropic/claude-opus-5-5@max", declared["route"]) is None

    provider._parent_route = staticmethod(lambda _cfg: ("anthropic", "claude-sonnet-5-5"))
    route = _clean_route(
        provider.name, provider.route_task({"route": "anthropic/claude-opus-5-5@max"}, config)
    )
    assert route["provider"] == "anthropic"
    assert route["model"] == "claude-opus-5-5"
    assert route["reasoning_effort"] == "max"
    _check_backend_effort(0, route, config)  # must not raise

    unrouted = _clean_route(provider.name, provider.route_task({}, config))
    assert unrouted == {}  # inherits the delegating agent's route


@pytest.mark.skipif(not CORE, reason="needs Hermes core importable")
def test_unrouted_task_inherits_the_parent():
    DynamicDelegationProvider = _pkg.provider.DynamicDelegationProvider
    assert DynamicDelegationProvider().route_task({}, {"backend": "dynamic"}) == {}


@pytest.mark.skipif(not CORE, reason="needs Hermes core importable")
def test_model_map_named_route_shortcut_is_honoured():
    DynamicDelegationProvider = _pkg.provider.DynamicDelegationProvider
    config = {
        "backend": "dynamic",
        "backends": {
            "model_map": {
                "options": [
                    {
                        "name": "opus55",
                        "description": "x",
                        "provider": "anthropic",
                        "model": "claude-opus-5-5",
                        "reasoning_effort": "high",
                    }
                ]
            }
        },
    }
    route = DynamicDelegationProvider().route_task({"route": "opus55"}, config)
    assert route["provider"] == "anthropic" and route["reasoning_effort"] == "high"
