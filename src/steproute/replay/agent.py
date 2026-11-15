"""The agent scaffold: drives one model through one task in a multi-step bash loop.

``BashAgent`` is a small loop with mini-SWE-agent 2.x's step shape (``docs/REPLAY.md``,
decision 1, says why it is ours and not mini-SWE-agent itself):

* a system message and one task message, then a linear history that only ever grows (every
  step's prompt is the previous prompt plus new messages, which is what prefix caching
  rewards and what the cost estimate assumes);
* one structured tool, ``bash``, called through the OpenAI chat-completions ``tools`` field;
* each command runs in a fresh subshell in the task's workspace, with a timeout, and its
  output comes back as a ``tool`` message, cut to its head and tail above 10,000 characters;
* the run ends when a command's output starts with ``COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT``
  (the rest of that output is the submitted patch), at the step limit, or after three replies
  in a row without a usable tool call;
* a reply whose tool call the provider itself could not parse (``ChatReply.unreadable``: Ollama
  fails the request instead of returning it) is handled as a reply with invalid JSON arguments;
* tool-call arguments that are not valid JSON are quoted back in the format error, with the
  parser's error, and never kept in the history (the same for every provider).

The workspace also offers an ``apply_patch`` command (``steproute.replay.apply_patch``); the
prompt does not mention it.

The scaffold only sees two small interfaces: a ``ChatClient`` (one model call; the runner's
client enforces every limit and cap) and a ``Workspace``. Another scaffold can be swapped in
by implementing ``Scaffold``. Standard library only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from steproute.replay.workspace import Workspace

__all__ = [
    "BASH_TOOL",
    "SUBMIT_MARKER",
    "AgentResult",
    "BashAgent",
    "ChatClient",
    "ChatReply",
    "RejectedCall",
    "RunStopped",
    "Scaffold",
    "invalid_json_error",
]

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

BASH_TOOL: Mapping[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute"}
            },
            "required": ["command"],
        },
    },
}

SYSTEM_TEMPLATE = (
    "Run {nonce}.\n"
    "You are a helpful assistant that can interact with a computer shell to solve programming "
    "tasks."
)

INSTANCE_TEMPLATE = """<pr_description>
Consider the following PR description:
{task}
</pr_description>

<instructions>
# Task Instructions

You are a software engineer working in a shell. Make changes to non-test files in the current
directory so that the issue in the PR description is fixed, in a way that is general and
consistent with the codebase.

This is an interactive process: think, issue at least one command, see the result, then issue
your next command(s).

## For each response

1. Explain your reasoning in a short THOUGHT section.
2. Call the `bash` tool one or more times. Every response MUST contain at least one tool call.

## Boundaries

- MODIFY: regular source files in the working directory.
- DO NOT MODIFY: tests and configuration files (pyproject.toml, setup.cfg, ...).

## Recommended workflow

1. Find and read the relevant code.
2. Write a script that reproduces the issue.
3. Edit the source to fix it.
4. Run your script again to check the fix, then try edge cases.

## Command execution

- Every command runs in a new subshell: `cd` and environment changes do not persist. Prefix
  a command with them when you need them (`cd sub && ...`).
- Use non-interactive flags; do not use editors or pagers that wait for input.
- The output of a long command is cut to its beginning and end; use head, tail, grep or sed.

## Submission

When you are done, submit your changes as a git patch, in this order and as SEPARATE
commands:

1. `git diff -- path/to/file1 path/to/file2 > patch.txt`, listing only the source files you
   changed to fix the issue (no tests, reproduction scripts or helpers). Do not commit.
2. Inspect patch.txt: its headers must show `--- a/` and `+++ b/` paths.
3. Submit with exactly: `echo {marker} && cat patch.txt`

After submitting you cannot continue working on the task.
</instructions>"""

INVALID_JSON = "the arguments are not valid JSON"

#: The most characters of rejected arguments quoted back to the model.
QUOTE_MAX_CHARS = 2000

#: The shape the arguments must have, said after invalid JSON.
ARGUMENTS_SHAPE = 'The arguments must be one JSON object whose "command" is a single string.'

FORMAT_ERROR = (
    "Tool call error: {error}\n\n"
    "Every response must call the `bash` tool at least once, with arguments "
    '{{"command": "<your command>"}}. If you have finished, submit as the first message '
    "describes."
)


class RunStopped(Exception):
    """Raised by a ``ChatClient`` to end the run: ``status`` is a journal status
    (``run_cap``, ``context_limit``, ``wall_time``, ``provider_error``, ...)."""

    def __init__(self, status: str, reason: str) -> None:
        super().__init__(f"{status}: {reason}")
        self.status = status
        self.reason = reason


@dataclass(frozen=True)
class RejectedCall:
    """Tool-call arguments a provider could not parse, as its error answer quoted them.

    ``length``: the length of the arguments the provider sent, when the proxy cut them."""

    arguments: str
    parser_error: str
    length: int | None = None


@dataclass(frozen=True)
class ChatReply:
    """The assistant message of one call (OpenAI shape: ``content`` and ``tool_calls``).

    ``unreadable``: the model replied with a tool call the provider could not parse and failed
    the request, so the reply itself is lost (``message`` is empty). The scaffold treats it as
    a tool call with invalid JSON arguments. ``rejected``: those arguments and the provider's
    parser error, when its answer quoted them."""

    message: Mapping[str, Any]
    finish_reason: str | None = None
    unreadable: bool = False
    rejected: RejectedCall | None = None


def invalid_json_error(
    arguments: str | None, parser_error: str | None, length: int | None = None
) -> str:
    """The format error for tool-call arguments that are not valid JSON: the arguments the
    model sent (cut to ``QUOTE_MAX_CHARS``), the parser's error and the shape they must have.
    The same text for every provider: the parser error is the provider's when it rejected the
    call (Ollama), else Python's ``json`` error. ``arguments = None``: the provider gave none."""
    if arguments is None:
        return f"{INVALID_JSON} (the provider did not return them). {ARGUMENTS_SHAPE}"
    total = max(len(arguments), length or 0)
    shown = arguments[:QUOTE_MAX_CHARS]
    lines = [
        f"{INVALID_JSON}.",
        f"Parser error: {parser_error or 'unknown'}",
        "Your arguments were:",
        "<arguments>",
        shown,
        "</arguments>",
    ]
    if total > len(shown):
        lines.append(f"(cut: the first {len(shown):,} of {total:,} characters are shown)")
    lines.append(ARGUMENTS_SHAPE)
    return "\n".join(lines)


class ChatClient(Protocol):
    def complete(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> ChatReply: ...


@dataclass
class AgentResult:
    #: ``submitted``, ``no_submission`` (gave up: repeated replies without a tool call) or
    #: ``step_limit``.
    exit: str
    submission: str | None
    steps: int
    messages: list[dict[str, Any]] = field(default_factory=list)


class Scaffold(Protocol):
    name: str

    def run(
        self, task_text: str, workspace: Workspace, client: ChatClient, *, nonce: str
    ) -> AgentResult: ...


def _observation(output: str, returncode: int | None, limit: int) -> str:
    rc = "timeout" if returncode is None else str(returncode)
    if len(output) <= limit:
        return f"<returncode>{rc}</returncode>\n<output>\n{output}</output>"
    half = limit // 2
    return (
        f"<returncode>{rc}</returncode>\n<warning>\nThe output of your last command was too "
        "long. Use head, tail, grep or sed to look at a smaller part.\n</warning>\n"
        f"<output_head>\n{output[:half]}\n</output_head>\n"
        f"<elided_chars>\n{len(output) - 2 * half} characters elided\n</elided_chars>\n"
        f"<output_tail>\n{output[-half:]}\n</output_tail>"
    )


class BashAgent:
    """The default scaffold (see the module docstring)."""

    #: ``steproute-bash-2`` since issue 75: invalid JSON arguments are quoted back to the model
    #: and left out of the history, and the workspace has an ``apply_patch`` command. Runs
    #: finished before keep ``steproute-bash-1`` in the journal.
    name = "steproute-bash-2"

    def __init__(
        self,
        *,
        step_limit: int = 250,
        command_timeout_s: float = 60.0,
        max_observation_chars: int = 10_000,
        max_format_errors: int = 3,
    ) -> None:
        self.step_limit = step_limit
        self.command_timeout_s = command_timeout_s
        self.max_observation_chars = max_observation_chars
        self.max_format_errors = max_format_errors

    def initial_messages(self, task_text: str, nonce: str) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": SYSTEM_TEMPLATE.format(nonce=nonce)},
            {
                "role": "user",
                "content": INSTANCE_TEMPLATE.format(task=task_text, marker=SUBMIT_MARKER),
            },
        ]

    def run(
        self, task_text: str, workspace: Workspace, client: ChatClient, *, nonce: str
    ) -> AgentResult:
        messages = self.initial_messages(task_text, nonce)
        steps = 0
        format_errors = 0
        while True:
            if steps >= self.step_limit:
                return AgentResult("step_limit", None, steps, messages)
            reply = client.complete(messages, [BASH_TOOL])
            steps += 1
            if reply.unreadable:
                # The reply is lost. An assistant turn with empty content and no tool call keeps
                # the roles alternating (user or tool, assistant, user), the same shape an empty
                # reply has; the format error after it quotes the invalid arguments when the
                # provider's answer gave them.
                msg: dict[str, Any] = {"role": "assistant", "content": ""}
                commands: list[tuple[str, str]] = []
                rej = reply.rejected
                error: str | None = (
                    invalid_json_error(rej.arguments, rej.parser_error, rej.length)
                    if rej is not None
                    else invalid_json_error(None, None)
                )
            else:
                msg = self._message(reply.message)
                commands, error, invalid = self._parse(msg.get("tool_calls", []))
                if invalid:
                    # Arguments that are not valid JSON do not go back into the history: a
                    # strict provider rejects the next request over them (Ollama; LiteLLM's
                    # Anthropic conversion parses them). The format error quotes them instead,
                    # so the turn has the shape of the unreadable reply above, with its text.
                    msg = {"role": "assistant", "content": msg.get("content") or ""}
            messages.append(msg)
            if error is not None:
                format_errors += 1
                for c in msg.get("tool_calls", []):
                    # Every tool call must be answered, or the next request is invalid.
                    messages.append(
                        {"role": "tool", "tool_call_id": c["id"], "content": "not executed"}
                    )
                messages.append({"role": "user", "content": FORMAT_ERROR.format(error=error)})
                if format_errors >= self.max_format_errors:
                    return AgentResult("no_submission", None, steps, messages)
                continue
            format_errors = 0
            for call_id, command in commands:
                r = workspace.execute(command, timeout_s=self.command_timeout_s)
                lines = r.output.lstrip().splitlines(keepends=True)
                if lines and lines[0].strip() == SUBMIT_MARKER and r.returncode == 0:
                    messages.append(
                        {"role": "tool", "tool_call_id": call_id, "content": "submitted"}
                    )
                    return AgentResult("submitted", "".join(lines[1:]), steps, messages)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": _observation(r.output, r.returncode, self.max_observation_chars),
                    }
                )

    @staticmethod
    def _message(reply: Mapping[str, Any]) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": "assistant", "content": reply.get("content")}
        calls = reply.get("tool_calls") or []
        if calls:
            msg["tool_calls"] = [
                {
                    "id": c.get("id"),
                    "type": "function",
                    "function": {
                        "name": c.get("function", {}).get("name"),
                        "arguments": c.get("function", {}).get("arguments") or "",
                    },
                }
                for c in calls
                if isinstance(c, Mapping)
            ]
        return msg

    @staticmethod
    def _parse(
        calls: Sequence[Mapping[str, Any]],
    ) -> tuple[list[tuple[str, str]], str | None, bool]:
        """The commands, or the format error; the flag says the error is invalid JSON."""
        if not calls:
            return [], "no tool call found in the response", False
        out = []
        for c in calls:
            fn = c.get("function") or {}
            if fn.get("name") != "bash":
                return [], f"unknown tool {fn.get('name')!r}", False
            raw = fn.get("arguments") or ""
            try:
                args = json.loads(raw)
            except ValueError as exc:
                return [], invalid_json_error(str(raw), str(exc)), True
            if not isinstance(args, dict) or not isinstance(args.get("command"), str):
                return [], "missing 'command' argument", False
            out.append((str(c.get("id")), args["command"]))
        return out, None, False
