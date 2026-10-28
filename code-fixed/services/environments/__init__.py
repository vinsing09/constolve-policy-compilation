"""Pluggable environments for journey eval.

Importing this package eagerly imports every concrete environment module so
the `@register_env(...)` decorators run and populate `ENV_REGISTRY`. Without
this, `ENV_REGISTRY.get("tau_bench")` from the orchestrator silently fell
back to `StubEnv` and every tool call returned `no_stub_for_<tool>`.

To add a new env: create the module, decorate the class with `@register_env`,
and add one import line below.
"""
from services.environments import base  # noqa: F401
from services.environments import stateless_env  # noqa: F401
from services.environments import stub_env  # noqa: F401
from services.environments import http_env  # noqa: F401
from services.environments import tau_bench_env  # noqa: F401
from services.environments import tau3_bench_env  # noqa: F401
from services.environments import compiled_env  # noqa: F401
