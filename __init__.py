"""dynamic-delegation: name provider + model + thinking level per subagent, at call time.

Registers one delegation backend (`delegation.backend: dynamic`) and one tool
(`dynamic_delegation_models`). Built on the same seam as the official advanced-delegation plugin
(`ctx.register_delegation_provider`), and reads the official plugin's `model_map` options as named
shortcuts, so `hermes delegation config` and the Desktop route editor stay useful.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _live_settings() -> dict:
    """The `delegation.backends.dynamic` block of the active profile, read per tool call."""
    try:
        from hermes_cli.config import load_config_readonly as load
    except ImportError:  # older cores
        from hermes_cli.config import load_config as load
    try:
        config = load() or {}
    except Exception:
        return {}
    delegation = config.get("delegation") if isinstance(config, dict) else None
    backends = delegation.get("backends") if isinstance(delegation, dict) else None
    block = backends.get("dynamic") if isinstance(backends, dict) else None
    return dict(block) if isinstance(block, dict) else {}


def register(ctx) -> None:
    try:
        from .provider import DynamicDelegationProvider
        from .discovery import SCHEMA, TOOLSET, make_handler
    except ModuleNotFoundError as exc:
        if exc.name != "agent.delegation_provider":
            raise
        logger.warning(
            "dynamic-delegation needs a Hermes core with pluggable delegation providers "
            "(agent.delegation_provider, Hermes 0.22.5+); update Hermes. The plugin is inactive.")
        return
    ctx.register_delegation_provider(DynamicDelegationProvider())
    try:
        ctx.register_tool(name=SCHEMA["name"], toolset=TOOLSET, schema=SCHEMA,
                          handler=make_handler(_live_settings),
                          description=SCHEMA["description"], emoji="🧭")
    except Exception as exc:  # a tool that cannot register must not take the backend down
        logger.warning("dynamic-delegation: could not register %s (%s); the route parameter still works.",
                       SCHEMA["name"], exc)
