"""Parameterized LLM-backed user simulator.

Contract:
- reply(history, task) returns a {message, done} pair.
- Persona fragment is inserted into the SYSTEM prompt.
- `wont_volunteer` facts are NOT surfaced unless the agent asks directly.
- `will_reveal_if_asked` facts are revealed when the agent asks.
- The sim ends the conversation when the goal is met OR the agent declines.

The won't_volunteer enforcement is what keeps multi-turn evaluation honest —
a sim that dumps all facts upfront defeats the point of a journey test.
"""
from __future__ import annotations

import json
import os
from typing import Any

import litellm
from pydantic import BaseModel


_SYSTEM_TEMPLATE = """You are simulating a user talking to an AI agent.

Your goal: {goal}
Facts you know:
{known_facts}
Facts you WILL NOT volunteer (only reveal if the agent asks specifically):
{wont_volunteer}
Facts you WILL reveal if the agent asks:
{will_reveal_if_asked}

Persona:
{persona_fragment}
{task_instructions}{acceptance}{user_script}

Rules:
1. Stay in character. Never break the fourth wall, never mention you are an
   AI or a simulation.
2. Do NOT invent facts that aren't in the lists above. If asked something
   you don't know, say so naturally.
3. Do NOT volunteer anything from the "WILL NOT volunteer" list. Only share
   a fact from that list if the agent asks a question that clearly
   requires it.
4. Respond with one short message appropriate for a real user.
5. AGREEING THE PROBLEM IS FIXED and THE CONVERSATION BEING OVER are two
   different things, and you decide them separately.
   - Do NOT agree the problem is fixed while any condition in "What you will
     accept" is unmet, even if the agent says it is fixed. That section
     overrides your own judgement.
   - Set `done: true` ONLY when there is genuinely nothing left for you to do:
     the agent has handed you to a specialist or another team, has told you it
     cannot do this, or has asked you to wait with NO further step for you to
     take. You are ending the CALL, not agreeing you were helped.
   - NEVER set `done: true` in the same reply where the agent has asked you a
     question or asked you to try something. Do the thing, or answer, and
     report what happened. An agent saying "that should have fixed it, please
     try again" is asking you to try again — so try, and say whether it worked.
     Hanging up there ends the call while the problem is still there, which is
     the one thing a real customer would not do.
   - If you have already said you will wait and the agent only repeats that you
     should wait, set `done: true`. Repeating the same exchange is not progress.
{script_overrides}
{user_tools}

{observations}
Respond in strict JSON:
{{"message": "<your reply, 1–3 sentences>", "done": <true|false>{actions_field}}}
No prose, no code fences."""


_OBSERVATIONS_BLOCK = """
WHAT YOU JUST SAW ON YOUR DEVICE. These are the real results of the actions you took a
moment ago, this turn:

{results}

You may act AGAIN now if what you saw calls for it, or speak. If you speak, describe what
you actually saw above — never what you assume or hope. If a result did not show the
updated status you expected, run the check again rather than guessing.
"""


_USER_TOOLS_BLOCK = """
YOU CAN ACT, NOT ONLY SPEAK. These are things only YOU can do — the agent cannot
do them for you and will have to talk you through them:

{tools}

6. When the agent asks you to do one of these, and you are willing, put it in
   `actions` AND say what you did. Perform an action only when asked, or when your
   own instructions tell you to.
   NEVER say you have done something unless it is in `actions` this turn. Saying "I
   turned it off" without the action means nothing changed on your phone, and the
   agent will keep troubleshooting a fault you told them was fixed. If you are not
   performing it, say so plainly instead.
7. You judge whether the problem is fixed by what you OBSERVE after acting, not by
   what the agent tells you. If it is not fixed, say so and stay on the call."""


class UserReply(BaseModel):
    message: str
    done: bool = False
    # Things the user DID this turn. τ³ telecom requires user-side actions in 94 of
    # 114 tasks — the agent cannot toggle data saver on someone else's phone, it has
    # to instruct them. Without this the user can only talk, which caps our telecom
    # journeys at a median of 8 turns against τ³'s 20 and makes the whole
    # fault-diagnosis archetype inexpressible. Empty for every journey that declares
    # no user tools, so nothing existing changes.
    actions: list[dict] = []


class UserSim:
    def __init__(self, model: str | None = None, temperature: float = 0.7):
        self.model = model or os.getenv("USER_SIM_MODEL", os.getenv("DEFAULT_MODEL", "claude-sonnet-4-6"))
        self.temperature = temperature

    async def reply(self, history: list[Any], task: Any,
                    observations: list[dict] | None = None,
                    rounds: list[dict] | None = None) -> UserReply:
        # Accept either dict-shaped task (from DB) or Pydantic TestGridCase.
        tget = (lambda key, default=None: getattr(task, key, default)) if not isinstance(task, dict) else (lambda key, default=None: task.get(key, default))

        # A journey may carry its persona TEXT rather than a registry id. Compiled
        # personas are built per domain — `handset_in_hand_no_service__plain` — and are
        # not in BUILT_IN_BY_ID, so the lookup below raised KeyError and quietly
        # substituted the cooperative default. The whole persona axis was disconnected:
        # a 16-persona arm and a fixed-persona arm returned the same 10.0% because they
        # were the same experiment. An unknown id is a REASON TO USE THE SUPPLIED TEXT,
        # never a reason to silently pick someone else.
        persona_fragment = (tget("persona_fragment") or "").strip()
        if not persona_fragment:
            from services.personas import get_persona
            persona_id = tget("persona_id") or "cooperative"
            try:
                persona = get_persona(persona_id)
                persona_fragment = persona.system_prompt_fragment
            except KeyError:
                persona_fragment = "You are a cooperative user. Answer clearly."

        goal = tget("goal") or ""
        known_facts = tget("known_facts") or {}
        # §30.27: the behavioural script. Rules 2 and 5 above are correct for a
        # cooperative journey and make the DECEPTION and PRESSURE categories
        # impossible — rule 2 forbids asserting anything unsupported, rule 5 ends the
        # conversation on the agent's first refusal. A script overrides them explicitly.
        from services import user_script as _us
        script = tget("user_script") or {}
        wont_volunteer = tget("wont_volunteer") or []
        will_reveal = tget("will_reveal_if_asked") or []

        user_tools = ((tget("world_setup") or {}).get("user_tools")
                      or tget("user_tools") or [])
        tools_block = ""
        if user_tools:
            listed = "\n".join(
                f"  - {t.get('name')}: {str(t.get('description') or '')[:160]}"
                for t in user_tools if isinstance(t, dict) and t.get("name"))
            tools_block = _USER_TOOLS_BLOCK.format(tools=listed)

        # WHAT THE PERSON WILL ACCEPT, REFUSE OR TAKE INSTEAD.
        # Measured across all three tau3 domains, a conditional branch is the only
        # element present in the majority of tasks everywhere, and it is what makes a
        # capable agent that never read the policy behave differently from one that did.
        # It has to reach the simulator or it changes nothing — the persona axis was
        # silently disconnected in exactly this way earlier today.
        accept_lines = [str(x) for x in (tget("acceptance") or []) if str(x).strip()]
        acceptance = ("\nWhat you will accept:\n"
                      + "\n".join(f"- {x}" for x in accept_lines) + "\n"
                      if accept_lines else "")

        # tau3's four-layer instruction block, when the case carries one. It holds the
        # grounding rule — "never make up the results of a tool call" — which is what
        # makes the customer RUN a check instead of reciting the answer. Measured on 630
        # transcripts: without it the customer answered from its script and 63% of graded
        # criteria (the user-side ones) were unsatisfiable.
        # WHAT THE LAST ACTIONS RETURNED. Without this the simulator wrote its message
        # before its own actions ran, so it could only assert; tau3's user reads each
        # result and decides again inside the same turn.
        obs_block = ""
        if observations:
            listed = "\n".join(
                f"  - {o.get('tool')}: "
                f"{json.dumps(o.get('result'), default=str)[:220] if 'result' in o else o.get('error')}"
                for o in observations if isinstance(o, dict))
            obs_block = _OBSERVATIONS_BLOCK.format(results=listed)
        ti = str(tget("task_instructions") or "").strip()
        task_instructions = f"\n{ti}\n" if ti else ""
        system = _SYSTEM_TEMPLATE.format(
            observations=obs_block,
            task_instructions=task_instructions,
            goal=goal,
            known_facts=json.dumps(known_facts, indent=2),
            wont_volunteer="\n".join(f"- {x}" for x in wont_volunteer) or "(none)",
            will_reveal_if_asked="\n".join(f"- {x}" for x in will_reveal) or "(none)",
            persona_fragment=persona_fragment,
            acceptance=acceptance,
            user_script=_us.to_prompt(script),
            script_overrides=_us.overrides_default_rules(script),
            user_tools=tools_block,
            actions_field=(', "actions": [{{"tool": "<name>", "params": {{}}}}]'
                           if user_tools else ""),
        )

        messages = [{"role": "system", "content": system}]
        for turn in history:
            role = getattr(turn, "role", None) if not isinstance(turn, dict) else turn.get("role")
            content = getattr(turn, "content", "") if not isinstance(turn, dict) else turn.get("content", "")
            if role == "agent":
                # The completion model is playing the customer.  Therefore prior
                # agent utterances are input from its counterparty (`user`) and prior
                # customer utterances are its own (`assistant`) history.
                messages.append({"role": "user", "content": content})
            elif role == "user":
                messages.append({"role": "assistant", "content": content})
        # WHAT THIS TURN HAS ALREADY DONE, as conversation rather than as instructions.
        #
        # The observation loop re-asked the model each round with the SAME history and the
        # same "reply as the user now" prompt, changing only a block inside the system
        # message. The model had no record of having already spoken or acted, so it
        # answered the identical question identically: measured 2026-08-21, 48% of our
        # acting turns repeated an action against tau3's 3%, and 38% of toggle turns
        # called the toggle an EVEN number of times, returning the world to exactly where
        # it started while the transcript recorded the action as performed.
        #
        # tau3 has no such problem because its user simulator keeps its own assistant
        # messages and the tool results in one growing message list — an ordinary
        # tool-calling loop. This makes ours the same shape.
        for r in (rounds or []):
            said = str(r.get("message") or "").strip()
            acts = r.get("actions") or []
            if acts:
                names = ", ".join(str(a.get("tool")) for a in acts if isinstance(a, dict))
                messages.append({"role": "assistant",
                                 "content": f"{said}\n[you performed: {names}]"})
            elif said:
                messages.append({"role": "assistant", "content": said})
            res = r.get("results") or []
            if res:
                listed = "\n".join(
                    f"  - {o.get('tool')}: "
                    f"{json.dumps(o.get('result'), default=str)[:200] if 'result' in o else o.get('error')}"
                    for o in res if isinstance(o, dict))
                messages.append({"role": "user",
                                 "content": f"Those actions have ALREADY been carried out "
                                            f"on your device. They returned:\n{listed}\n"
                                            f"Do not repeat them. Either take a DIFFERENT "
                                            f"action if one is needed, or speak to the "
                                            f"agent about what you saw."})
        # Prompt the sim for the next user turn.
        messages.append({"role": "user", "content": "Reply as the user now (JSON only)."})

        try:
            response = await litellm.acompletion(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw = (response.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:].lstrip()
            data = json.loads(raw)
            acts = data.get("actions") or []
            # Only accept actions the journey actually granted. A sim that invents a
            # tool would otherwise mutate the world through a name nobody declared.
            allowed = {t.get("name") for t in user_tools if isinstance(t, dict)}
            acts = [a for a in acts if isinstance(a, dict)
                    and a.get("tool") in allowed]
            return UserReply(message=str(data.get("message", "")),
                             done=bool(data.get("done", False)), actions=acts)
        except Exception as e:
            return UserReply(
                message=f"(user-sim error: {type(e).__name__})",
                done=True,
            )
