# Internals

The deep detail behind the README's one diagram. Read this if you want to change the plugin, review
it, or understand why it is allowed to exist at all.

## The call direction

The plugin is called by `delegate_task`. It never calls the subagent machinery itself.

```
delegate_task handler  (tools/delegate_tool.py)
   ├─ :820  schema_extension()        -> provider.task_parameters()   [plugin]
   └─ :555  _routes_for()
              └─ resolve_task_routes() -> provider.route_task()        [plugin]
                     route dict returns to core, which builds and runs the child
```

`delegate_task` is a core tool and is not overridden. Overriding a core tool name takes
`override=True` plus the operator gate `plugins.entries.<id>.allow_tool_override`
(`hermes_cli/plugins_context.py:277-286`), and this plugin asks for neither. The only tool it
registers is `dynamic_delegation_models`.

## The seam we did not use

There is a lower seam: the public Subagent Lifecycle API,
`ctx.subagent_lifecycle.launch(SubagentLaunchRequest(...))`. A plugin calling that bypasses
`delegate_task` entirely, which is how hermes-herald's `delegate_subagent` works. It is a worse fit
here for two reasons:

- `SubagentLaunchRequest` has 12 fields (`agent/subagent_lifecycle.py:48-61`) and no
  `reasoning_effort`, `service_tier`, `base_url`, `api_key` or `api_mode`. Per-task thinking level is
  unreachable through it. Adding it is open upstream as PR #140727.
- Bypassing `delegate_task` loses task groups, `output_schema`, progress reporting and live
  transcripts.

The delegation-provider seam keeps all of those, and reaches per-task effort through core's existing
`_build_child_agent(override_reasoning_effort=...)` parameter, documented as "a reasoning effort
that beats delegation.reasoning_effort" (`tools/delegate_tool.py:219-221`).

## Why a plugin can expose model choice at all

Core refuses to make `provider`, `model`, `base_url`, `api_key` and `api_mode` model-facing task
parameters. That list is `RESERVED_TASK_PARAMETERS` in `agent/delegation_provider.py`, enforced by
`_declared_parameters` in `tools/delegate_tool_routes.py:64-82`. Upstream states the reason plainly
(technium1, closing #88891): "the agent picks from your allowlist and never types a model id."

Core does not restrict a provider-declared parameter with any other name, and the only value check it
performs on one is `enum` (`_param_error`, `tools/delegate_tool_routes.py:100-104`). No type, pattern,
`minLength` or catalog check runs. So the selection travels as one free-form string that this plugin
decodes, and core validates the route dict it gets back exactly as it always has:

| Check | Where |
|---|---|
| parameter name not reserved, matches `^[a-z][a-z0-9_]{0,63}$` | `_declared_parameters` |
| enum membership, when the parameter declares one | `_param_error` |
| `reasoning_effort` parses | `_check_backend_effort` |
| `service_tier` is offered by the resolved route | `tools/delegate_tool_tuning.py` |
| provider resolves to credentials, else the task is refused | `_resolve_delegation_credentials` |
| route keys outside `ROUTE_KEYS` are dropped | `_clean_route` |

Identical routes resolve credentials once per batch call, and a task that fails routing is refused
alone while its siblings run.

## Inside `route_task()`

Two stages: build a catalog once per TTL, then resolve one string against it per task.

```
build_catalog()                      once per catalog_ttl_seconds
  config.yaml providers:<name>.models      hand-maintained; carries context length and vision
  custom:<base_url>#<fingerprint>          the live endpoint catalog, what the picker shows
  authenticated catalog providers          provider_models_cache.json, then the static catalog
      |
  apply_desktop_filter()               visible-models, known-models fall-through, model-presets
      |
  ProviderEntry list, cached, keyed on the settings that shape it

resolve(spec, catalog)               per task
  "fireworks glm-5p3-flash@high+fast"
      |
  parse_spec()      target "fireworks glm-5p3-flash", effort high, tier fast
  _split_target()   provider fireworks, model hint "glm-5p3-flash"
                    (prefix, suffix and multi-word labels all tried)
  _match_models()   ranked: exact 100 > token-boundary prefix 80 > all-tokens 70
                    > word-prefix 65 > substring 40 > fuzzy. Floor is 60.
      |
      +-- one winner above the floor -> Resolved(provider, model, effort, tier)
      +-- tie, or best below the floor -> RouteRefused with ranked candidates
      +-- provider absent from the catalog -> provider_is_routable(), core's own resolver
                                             decides; reported as unverified
```

`as_route()` turns the `Resolved` into the dict core consumes, and `label` carries the original
string so a result reports what you asked for, not only what you got.

### Why the scoring floor is 60

Below token-boundary matching, a short id gets absorbed by a longer one: substring matching let
`kimi-k3` resolve to `kimi-k3p1-pro`, which is a different model. Exact ids score 100, so a real
`kimi-k3` in the list always wins over the fuzzy neighbour. A tie at the top score refuses rather
than picking a model generation for the caller.

### Why the catalog is advisory

Core's credential resolver is the real gate: an unknown provider is refused per task. So a provider
the agent names that discovery missed is accepted when `resolve_runtime_provider` can authenticate
it, reported as `provider_kind: unverified` with model-id validation skipped, since there is no
catalog to validate against. Set `allow_providers` to make the catalog a hard allowlist.

## The Desktop filter's fall-through rule

Hidden means judged-once-and-not-shown. `hermes.desktop.known-models` snapshots every
`provider::model` key the user has been shown; a model absent from it arrived after the last
curation, so it falls through to visible. That mirrors the rule the Desktop store documents in
`apps/desktop/src/store/model-visibility.ts`, and keeps a new release from silently making a model
unroutable.

`visible-models` has no Python reader in Hermes core. It is written and consumed by the Desktop app
and persisted into `ui-preferences.json`, which is a machine-level file. This plugin is the only
consumer that applies it to subagent routing, and it does so on every surface, CLI included.

## Model discovery sources

| Source | Carries | Notes |
|---|---|---|
| `config.yaml providers.<name>.models` | ids, context length, vision | hand-maintained, goes stale |
| `provider_models_cache.json` key `custom:<base_url>#<fp>` | ids | the live endpoint catalog, what the Desktop picker shows |
| `provider_models_cache.json` key `<provider>` | ids | credential-fingerprinted, TTL 1h, 7d stale-while-revalidate |
| `static_catalog(provider)` | ids | Hermes' built-in list, no network |

`cached_provider_model_ids(..., non_blocking=True)` returns nothing once an entry is stale, which
would silently drop whole providers, so the cache file is read directly as a fallback.

## Tests

28 tests. The core-integration cases import real core functions
(`_declared_parameters`, `_param_error`, `_check_backend_effort`, `_clean_route`,
`DynamicDelegationProvider.route_task`) and skip when Hermes is not importable. The Desktop-filter
cases write a fixture `ui-preferences.json` into a temp `HERMES_HOME`, so they never read the
developer's live picker state.
