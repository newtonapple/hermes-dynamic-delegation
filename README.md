# hermes-dynamic-delegation

**Delegate any subagent to any model you have configured.**

Name the provider, the model and the thinking level per task, at the moment you delegate. One string
does it: `provider/model@effort`. Any model your Hermes can authenticate to is that far away, and a
spec the agent gets wrong is refused before a subagent starts.

```python
delegate_task(tasks=[
  {"goal": "Design the SQLite -> Postgres migration for the sessions table.",
   "route": "anthropic/claude-opus-5@max"},
  {"goal": "Rename FEATURE_X to FEATURE_Y at every call site under src/.",
   "route": "fireworks glm-5p3-flash@high+fast"},
  {"goal": "Summarize the attached log.",
   "route": "local-vllm coder@low"},   # fuzzy: -> local-vllm/qwen3-coder
  {"goal": "No route given."},         # inherits the delegating agent's model
])
```

## How it plugs in

Hermes 0.22.5 made subagent routing pluggable. We register a backend with
`ctx.register_delegation_provider`, and core asks it one question per task: which model?

```
delegate_task(tasks=[{ goal, context, route: "anthropic/claude-opus-5@max" }])
        |
        v
   Hermes core --- "which model for this task?" --->  route_task()
        ^                                                    |
        |<-------- provider / model / effort / tier ---------+
        v
   core builds and runs the subagent: toolsets, isolation, depth limits,
   prompt, context, progress, results. All native.
```

| | |
|---|---|
| We add | one task field, `route`, and one function, `route_task()` |
| We reuse | everything else, including the child's prompt and context handling |

`delegate_task` is not overridden. `delegation.backend` selects one active backend, so setting it to
`model_map` makes this plugin inert. Named routes from the official plugin's `model_map` list still
work as shortcuts, so `hermes delegation config` and the Desktop route editor keep their use.

The deep detail, including the call chain and the seam we deliberately avoided, is in
[docs/internals.md](docs/internals.md).

## Grammar

```
[provider/]model[@effort][+tier][+daybreak]
```

| Part | Values | Notes |
|---|---|---|
| `provider` | id, alias or label: `anthropic`, `openai-codex` (alias `chatgpt`), `openrouter`, `fireworks`, or any custom provider you configure, e.g. `local-vllm` | Optional. A bare model name is searched across every provider in the catalog |
| `model` | full id, or a fragment: `sonnet`, `opus 5`, `flash` | Ranked exact, token-prefix, all-tokens, substring. A weak match is refused with candidates, never guessed |
| `@effort` | `none, minimal, low, medium, high, xhigh, max, ultra` | Core validates the level. The wire clamps a level the model does not offer to the nearest weaker one |
| `+tier` | `+fast`, `+ultrafast` | Core refuses a tier the route cannot serve |
| `+daybreak` | OpenAI Daybreak trusted access | Requires `openai-codex`, gated by the official plugin's rules |

Separators are interchangeable. `anthropic/claude-sonnet-5`, `anthropic claude-sonnet-5` and
`anthropic:claude-sonnet-5` resolve identically, and a multi-word provider label works
(`Fireworks AI glm-5p3-flash`). A provider named last works too: `astra pi-codex`.

## Examples

Each block is the call the agent makes after you ask for a model in words. Tasks in one call run in
parallel, and each carries its own `route`.

A subagent starts blank. It sees its own `goal` and `context` and nothing else, so file paths,
constraints and acceptance criteria go in `context`, repeated per task when two tasks need the same
background.

**One subagent, exact model and thinking level.**

> Use Opus at max thinking to design the migration plan.

```python
delegate_task(tasks=[{
  "goal": "Design the migration plan for the sessions table.",
  "context": "Repo at ~/src/api. SQLite today, schema in src/db/schema.sql, 40 call sites in "
             "src/db/. Read schema.sql before proposing anything. Do not commit.",
  "route": "anthropic/claude-opus-5@max",
}])
```

**Fan out by difficulty.** Cheap model for the mechanical edit, strong model for the judgement call.

> Rename the flag across the repo on a cheap model, and have a strong model at high thinking review
> the public API for breakage. Run them in parallel.

```python
delegate_task(tasks=[
  {"goal": "Rename FEATURE_X to FEATURE_Y in every call site under src/.",
   "context": "The flag is declared in src/features.py. Tests live in tests/. Do not touch "
              "src/public/. Do not commit.",
   "route": "openrouter/google/gemini-3-flash-preview@low"},
  {"goal": "List breaking changes to the public API caused by the FEATURE_X rename.",
   "context": "Public API is everything exported from src/public/. Report file:line for each break "
              "and name the caller that breaks. Read-only, so change nothing.",
   "route": "anthropic/claude-opus-5@high"},
])
```

**Parallel research, one question per model.** This is a good way to find where models disagree, and
disagreement is where the interesting uncertainty lives.

> Ask three models the same research question in parallel and tell me where they disagree.

```python
delegate_task(tasks=[
  {"goal": "What are the tradeoffs of a looped transformer decoder? Cite sources.",
   "context": "Separate vendor claims from third-party measurements. Every non-obvious claim needs "
              "a URL. Under 600 words.",
   "route": "openai-codex/gpt-6-sol@high"},
  {"goal": "What are the tradeoffs of a looped transformer decoder? Cite sources.",
   "context": "Separate vendor claims from third-party measurements. Every non-obvious claim needs "
              "a URL. Under 600 words.",
   "route": "anthropic/claude-sonnet-5@high"},
  {"goal": "What are the tradeoffs of a looped transformer decoder? Cite sources.",
   "context": "Separate vendor claims from third-party measurements. Every non-obvious claim needs "
              "a URL. Under 600 words.",
   "route": "openrouter/deepseek/deepseek-v4-flash@medium"},
])
```

**Search a pile of documents.** One subagent per file, on a cheap local endpoint. The path belongs in
`context`, because the child has no shared history to read it from.

> Grep is not enough. Send each of these 12 PDFs to a local model in parallel and have each report
> every passage about contract termination, with page numbers.

```python
delegate_task(tasks=[
  {"goal": f"Report every passage in {path} about contract termination, with page numbers.",
   "context": f"File: {path}. Use the pdf skill to read it. Return a markdown list of "
              "page: quote, or the single word 'none'. Quote exactly, do not paraphrase.",
   "route": "local-vllm/qwen3-coder@low"}
  for path in files
])
```

Twelve parallel readers on a local endpoint costs nothing per token and finishes in about the time one
cloud call takes. That is the shape local models are good for: many small reads, not one deep
reasoning pass.

**Escalate only when the model was the problem.**

> Write the retry tests on a cheap model. If it fails because the model was too weak, redo it on a
> stronger one. Not if it failed because the spec was unclear.

```python
delegate_task(tasks=[{
  "goal": "Write tests for the backoff in src/net/retry.py.",
  "context": "Existing test style is in tests/net/. Cover jitter, the cap, and the overflow case.",
  "route": "openrouter/google/gemini-3-flash-preview@medium",
}])
# only on a failure whose cause is the model:
delegate_task(tasks=[{
  "goal": "Write tests for the backoff in src/net/retry.py.",
  "context": "Existing test style is in tests/net/. Cover jitter, the cap, and the overflow case. "
             "A previous attempt left these failures: <paste them>.",
  "route": "anthropic/claude-opus-5@high",
}])
```

**A custom provider, named by fragment.**

> Use the coder model on my box for this.

```python
delegate_task(tasks=[{
  "goal": "Format every JSON file under fixtures/.",
  "context": "Two-space indent, keys sorted, no trailing newline. Do not touch fixtures/golden/.",
  "route": "local-vllm coder",
}])
```

**What comes back.** Every routed result reports the route it ran on, so a wrong guess is visible
instead of silent.

```json
{"route": {"backend": "dynamic", "label": "anthropic/claude-opus-5@max",
           "provider": "anthropic", "model": "claude-opus-5",
           "reasoning_effort": "max", "service_tier": "normal"}}
```

**Refusals are the point.** Core validates the provider and the thinking level, and never the model
id. This plugin validates the model id, so a typo fails as a tool error before a subagent starts,
with candidates attached.

```
Task 0: 'luna 6' is not a confident match on openai-codex.
        Candidates: openai-codex/gpt-6-luna, openai-codex/gpt-5.6-luna
```

## Finding valid model ids

The agent calls `dynamic_delegation_models` to check a spec before spending a subagent on it.

| action | returns |
|---|---|
| `list` | every routable provider with model counts and aliases |
| `models` (`provider`) | model ids for one provider, with context length and vision |
| `search` (`query`) | fuzzy matches across all providers, as canonical specs |
| `check` (`route`) | dry run: the route a spec produces, the levels that model offers, and a note when the request would be clamped |
| `refresh` | rebuild the catalog |

## Install

```bash
hermes plugins install <path-or-repo> --enable
hermes config set delegation.backend dynamic
```

Plugins load per process, so a running gateway never picks up a new one. Restart it, then start a new
session: Hermes builds the `delegate_task` schema when a session starts, and a running chat keeps the
tool set it was built with.

## The Desktop model filter

Routable models are the ones you show in the Desktop model picker, not the full provider catalog. A
model you hid is refused, and the message says why. A provider with nothing visible drops out of the
list the agent sees. Your per-model picker settings (effort and service tier) apply when a spec names
no `@effort`.

The filter is a Desktop preference, and the CLI has none: `hermes model` and `/model` list the full
catalog. This plugin reads the same file from every surface, so subagent routing is filtered in the
CLI too. Set `respect_desktop_filter: false` to route over the full discovered catalog.

Matching is word-based. `kimi k3 pro` reaches `kimi-k3p1-pro`, and an exact id beats any fuzzy
neighbour. A fragment that ties, like `luna 6` matching both `gpt-6-luna` and `gpt-5.6-luna`, refuses
with both candidates rather than picking a generation for you.

## Config

Everything is optional. The block is `delegation.backends.dynamic`.

```yaml
delegation:
  backend: dynamic
  backends:
    dynamic:
      # default: "anthropic/claude-sonnet-5@high"   # route for tasks that name none
      allow_providers: []        # empty = every provider with credentials
      deny_providers: []
      extra_providers: []        # force-include provider ids discovery misses
      catalog_ttl_seconds: 300
      respect_desktop_filter: true
      aliases:
        deep: "anthropic/claude-fable-5-2@max"
        local: "local-vllm/qwen3-coder@medium"
```

## Security posture

- The grammar has no slot for `base_url` or `api_key`. An agent can only reach providers you already
  authenticated. Core's credential resolver is the gate, and it refuses an unknown provider per task
  while siblings in the same batch still run.
- The catalog is advisory, not a gate. A provider the agent names that discovery missed is accepted
  when core's resolver can authenticate it, reported as `provider_kind: unverified` with model-id
  validation skipped. Set `allow_providers` to make it a real allowlist.
- The plugin writes nothing to disk, adds no hooks, and makes no network call from the schema path.

## Files

| Path | What it does |
|---|---|
| `plugin.yaml` | The manifest core reads: name, `requires_hermes: ">=0.22.5"`, `provides_tools`, and the description shown by `hermes plugins list` and the Desktop plugin card. |
| `__init__.py` | The entry point, and the only file core imports. Core calls its `register(ctx)`, which registers the backend and the tool. Required: the loader skips a plugin directory with no `__init__.py`, so without it the directory is not a plugin. |
| `provider.py` | The backend class. `task_parameters()` adds the `route` field to the schema, `route_task()` turns a spec into a route dict, `get_catalog()` caches the catalog. |
| `catalog.py` | The catalog and the resolver: which providers and models exist, which ones your picker shows, and how a fuzzy spec maps to exactly one of them. Most of the logic lives here. |
| `discovery.py` | The `dynamic_delegation_models` tool: `list`, `models`, `search`, `check`, `refresh`. |
| `tests/test_dynamic.py` | 28 tests. The core-integration cases skip when Hermes is not importable. |
| `docs/internals.md` | The deep dive: call chain, the seam we avoided, resolver scoring, filter semantics. |
| `README.md`, `LICENSE` | This file, and MIT. |

Installed as `~/.hermes/plugins/dynamic-delegation/`, minus `tests/` and `docs/`. The runtime copy is a
plain file copy, not a symlink, so an iCloud eviction cannot empty a module behind a running process.

## Limits

- `ultra` passes core's validator and is Hermes-internal, the Codex app-server harness mode. Most wires
  clamp it to `max`.
- A provider with no discoverable model list is not offered. Custom providers read `config.yaml` plus
  the endpoint catalog cache; `extra_providers` forces an id in.
- Route changes apply to new sessions. A running chat's tool schema is part of its prompt cache.

## Tests

```bash
python -m pytest -q
```

## License

MIT. See [LICENSE](LICENSE).
