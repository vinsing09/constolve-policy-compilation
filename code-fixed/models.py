from __future__ import annotations
from enum import Enum
from typing import Any, Literal, Optional
from pydantic import BaseModel, field_validator


class ToolStub(BaseModel):
    response: dict
    latency_ms: int = 0
    simulate_failure: bool = False


class Assertion(BaseModel):
    id: str
    type: Literal[
        "tool_called", "tool_not_called", "param_contains", "output_contains",
        "max_latency_ms", "tool_sequence",
    ]
    tool: str | None = None
    param: str | None = None
    value: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def coerce_value_to_str(cls, v: Any) -> str | None:
        if v is None:
            return None
        if isinstance(v, str):
            return v
        return str(v)
    required: bool = True


class Agent(BaseModel):
    id: str
    name: str
    system_prompt: str
    tool_schemas: list[dict]
    created_at: str
    business_goal: str = ""
    desired_behaviors: list[str] = []
    hooks: dict | None = None  # {"pre_tool_use": [...], "post_tool_use": [...]}
    kind: str = "stateless"  # stateless | tool_using | multi_turn
    kind_confidence: float | None = None
    kind_reasons: list[str] = []
    # "parallel" = execute every tool call emitted in a turn; "serial" = only
    # the first (ReAct, matches tau-bench native). Default parallel for
    # backward-compat; tau-bench ingest opts into "serial".
    tool_execution_mode: Literal["serial", "parallel"] = "parallel"


class TestCase(BaseModel):
    id: str
    agent_id: str
    scenario: str
    input_text: str
    tool_stubs: dict[str, ToolStub]
    assertions: list[Assertion]
    locked: bool
    tags: list[str]
    created_at: str


class ToolCallRecord(BaseModel):
    tool_name: str
    params: dict
    response: dict
    latency_ms: int
    simulated_failure: bool
    sequence_order: int


class Trace(BaseModel):
    test_case_id: str
    tool_calls: list[ToolCallRecord]
    final_output: str
    total_latency_ms: int


class Contract(BaseModel):
    id: str
    agent_id: str
    obligations: list[str]
    tool_stubs: dict[str, "ToolStub"]
    version: int
    created_at: str


class EvalRun(BaseModel):
    id: str
    agent_id: str
    run_type: str
    started_at: str
    status: str


class EvalResult(BaseModel):
    id: str
    run_id: str
    test_case_id: str
    assertion_id: str
    passed: bool | None
    reason: str
    latency_ms: int
    result_type: str = "deterministic"


# ── V2 models ─────────────────────────────────────────────────────────────────

class AgentDraft(BaseModel):
    id: str
    name: str
    business_goal: str
    desired_behaviors: list[str]
    raw_system_prompt: str
    tool_schemas: list[dict]
    created_at: str


class AuditIssue(BaseModel):
    id: str
    severity: str  # "high" | "medium" | "low"
    description: str
    gap_type: str  # "missing" | "ambiguous" | "contradicts_goal"


class SuggestedFix(BaseModel):
    id: str
    issue_id: str
    description: str
    prompt_patch: str


class AuditReport(BaseModel):
    id: str
    agent_draft_id: str
    issues: list[AuditIssue]
    suggested_fixes: list[SuggestedFix]
    created_at: str


class AgentVersion(BaseModel):
    id: str
    agent_id: str
    version_number: int
    label: str
    system_prompt: str
    tool_schemas: list[dict]
    parent_version_id: str | None
    source: str  # "audit" | "manual" | "eval_driven"
    created_at: str
    schema_instance: dict | None = None


class AgentSchema(BaseModel):
    id: str
    agent_id: str
    schema_json: dict
    extracted_from_version_id: str
    human_edited: bool
    created_at: str


# ── test-grid-env-v1 (Phase 0 stubs; finalized in Phase 1/2) ─────────────────

class AgentKind(str, Enum):
    STATELESS = "stateless"
    TOOL_USING = "tool_using"
    MULTI_TURN = "multi_turn"


class InferredKind(BaseModel):
    kind: AgentKind
    confidence: float
    reasons: list[str]


class TestGridCase(BaseModel):
    id: str
    agent_id: str
    agent_version_id: Optional[str] = None
    contract_id: Optional[str] = None
    obligation_ids: list[str] = []
    kind: AgentKind = AgentKind.STATELESS
    scenario: str = ""
    # stateless fields (back-compat with existing TestCase)
    input_text: Optional[str] = None
    tool_stubs: Optional[dict[str, ToolStub]] = None
    assertions: Optional[list[Assertion]] = None
    # multi-turn fields (Phase 3)
    goal: Optional[str] = None
    known_facts: Optional[dict] = None
    wont_volunteer: list[str] = []
    will_reveal_if_asked: list[str] = []
    persona_id: Optional[str] = None
    world_seed: Optional[str] = None
    # §30.8 D16: the world that must pre-exist for this journey to be a test.
    # world_seed is a hash of (goal, rule, tool) and cannot describe one.
    world_setup: Optional[dict] = None
    # §30.28: claims / persistence / prohibitions / secondary_intent / constraints
    user_script: Optional[dict] = None
    success_criteria: Optional[list[dict]] = None
    coverage_tags: list[str] = []
    source: Literal["llm_generated", "trace_derived", "human_authored", "imported"] = "llm_generated"
    locked: bool = False
    tags: list[str] = []
    created_at: str = ""


class Criterion(BaseModel):
    """Journey-level success criterion — see Phase 4 verdict.evaluate."""
    id: str
    type: Literal["world_state_check", "behavior_check", "policy_check",
                  "answer_check"]
    description: str = ""
    # world_state_check
    jsonpath: Optional[str] = None
    expected: Optional[Any] = None
    # A threshold graded as equality punishes an agent that did BETTER than required:
    # a cause of "download_speed at least 'good'" became "== 'good'", so reaching
    # 'excellent' failed. `order` is the field's enum, which IS the ranking.
    comparator: Optional[str] = None   # "at_least" | "not_eq"; exact when absent
    order: Optional[list] = None
    # behavior_check / tool_called_with: which arguments the case actually grades.
    # None = all of them, [] = the tool name only. Free-text parameters belong in [].
    compare_args: Optional[list] = None
    # behavior_check
    rule: Optional[str] = None  # "tool_called" | "tool_not_called" | "max_turns" | "user_said_done"
    param: Optional[dict] = None
    # policy_check
    policy_doc: Optional[str] = None
    # answer_check (§30.30): COMPUTATION tasks — 7 of τ³'s 50 — require a correct
    # computed ANSWER (cheapest flight under constraints, total savings). No existing
    # type can assert on one: tool_called says a tool ran, policy_check says no rule was
    # broken, world_state_check inspects the database. An agent can satisfy all three and
    # still tell the user the wrong number.
    expected_answer: Optional[str] = None
    answer_tolerance: Optional[float] = None
    judge_name: Optional[str] = None  # e.g. "goal_achieved" | "hallucination" | custom


class CriterionResult(BaseModel):
    criterion_id: str
    passed: bool
    reason: str = ""


class Verdict(BaseModel):
    journey_id: str
    passed: bool
    results: list[CriterionResult]
    history_len: int = 0
    final_state: Optional[dict] = None

