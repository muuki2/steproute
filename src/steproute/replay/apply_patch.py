"""apply_patch: apply a patch in the "*** Begin Patch" format to files in the current directory.

The runner installs this file as the ``apply_patch`` command in every task workspace
(``steproute.replay.workspace``). Usage::

    apply_patch <<'PATCH'
    *** Begin Patch
    *** Update File: pkg/mod.py
    @@ def f():
    -    return 1
    +    return 2
    *** End Patch
    PATCH

or with the whole patch as one argument. The format:

* ``*** Begin Patch`` and ``*** End Patch`` enclose one or more file sections;
* ``*** Add File: PATH`` then the new file's lines, each starting with ``+``;
* ``*** Delete File: PATH``;
* ``*** Update File: PATH``, optionally ``*** Move to: NEW_PATH``, then hunks. A hunk starts
  with ``@@``, optionally followed by a line of the file (``@@ def f():``) that is found first;
  its lines start with a space (context), ``-`` (removed) or ``+`` (added). ``*** End of File``
  after a hunk anchors it at the end of the file.

A hunk is found by its context and removed lines, searching forward from the previous hunk;
when no exact match exists, lines that differ only in trailing whitespace match. Paths are
relative to the current directory and may not leave it. Nothing is written unless the whole
patch applies; on failure the command exits 1 and names the file and the first line that did
not match.

Written for old interpreters (task images run Python 3.6 and later): standard library only,
no f-strings, no variable annotations. Written for steproute from the format's description.
"""

import os
import re
import sys

BEGIN = "*** Begin Patch"
END = "*** End Patch"
ADD = "*** Add File: "
DELETE = "*** Delete File: "
UPDATE = "*** Update File: "
MOVE = "*** Move to: "
EOF_MARK = "*** End of File"
# A unified-diff range after "@@" ("@@ -1,3 +1,4 @@") carries no context line.
_RANGE = re.compile(r"^-\d+(,\d+)? \+\d+(,\d+)? @@")


class PatchError(Exception):
    pass


class Hunk:
    def __init__(self):
        self.headers = []
        self.lines = []  # (" ", "-" or "+", text)
        self.at_eof = False

    @property
    def old(self):
        return [text for tag, text in self.lines if tag != "+"]

    def replacement(self, found):
        """The new lines, given the file's lines the hunk matched: context lines keep the
        file's text (which may differ in trailing whitespace)."""
        out = []
        i = 0
        for tag, text in self.lines:
            if tag == " ":
                out.append(found[i])
            if tag != "+":
                i += 1
            if tag == "+":
                out.append(text)
        return out


class FileOp:
    def __init__(self, kind, path):
        self.kind = kind  # "add", "delete" or "update"
        self.path = path
        self.move_to = None
        self.lines = []  # "add": the new file's lines
        self.hunks = []  # "update"


def parse(text):
    """The file operations of a patch. Raises PatchError."""
    lines = text.replace("\r\n", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines or lines[0].strip() != BEGIN:
        raise PatchError("the patch must start with a line '" + BEGIN + "'")
    if lines[-1].strip() != END:
        raise PatchError("the patch must end with a line '" + END + "'")
    body = lines[1:-1]
    ops = []
    op = None
    hunk = None
    for number, line in enumerate(body, 2):
        if line.startswith(ADD):
            op = FileOp("add", line[len(ADD) :].strip())
            ops.append(op)
            hunk = None
        elif line.startswith(DELETE):
            op = FileOp("delete", line[len(DELETE) :].strip())
            ops.append(op)
            hunk = None
        elif line.startswith(UPDATE):
            op = FileOp("update", line[len(UPDATE) :].strip())
            ops.append(op)
            hunk = None
        elif op is None:
            if line.strip():
                raise PatchError(
                    "line {}: expected '{}', '{}' or '{}', got {!r}".format(
                        number, UPDATE.strip(), ADD.strip(), DELETE.strip(), line
                    )
                )
        elif op.kind == "add":
            if not line.startswith("+"):
                raise PatchError(
                    "line {}: every line of an added file must start with '+', got {!r}".format(
                        number, line
                    )
                )
            op.lines.append(line[1:])
        elif op.kind == "delete":
            if line.strip():
                raise PatchError(
                    "line {}: unexpected line after '{}{}': {!r}".format(
                        number, DELETE, op.path, line
                    )
                )
        elif line.startswith(MOVE) and not op.hunks:
            op.move_to = line[len(MOVE) :].strip()
        elif line.startswith("@@"):
            header = line[2:].strip()
            if _RANGE.match(header):
                header = ""
            if hunk is None or hunk.lines or hunk.at_eof:
                hunk = Hunk()
                op.hunks.append(hunk)
            if header:
                hunk.headers.append(header)
        elif line.strip() == EOF_MARK:
            if hunk is None:
                raise PatchError("line {}: '{}' before any hunk".format(number, EOF_MARK))
            hunk.at_eof = True
        else:
            if hunk is None or hunk.at_eof:
                hunk = Hunk()
                op.hunks.append(hunk)
            if line == "":
                # An empty context line whose leading space was dropped.
                hunk.lines.append((" ", ""))
            elif line[0] in " -+":
                hunk.lines.append((line[0], line[1:]))
            else:
                raise PatchError(
                    "line {}: a hunk line must start with ' ', '-' or '+', got {!r}".format(
                        number, line
                    )
                )
    if not ops:
        raise PatchError("the patch has no file sections")
    for op in ops:
        if not op.path:
            raise PatchError("a file section has no path")
        if op.kind == "update" and not op.hunks and op.move_to is None:
            raise PatchError("'" + UPDATE + op.path + "' has no hunks")
    return ops


def check_path(root, path):
    """The absolute path of ``path`` under ``root``. Raises PatchError for an absolute path or
    one that leaves ``root`` (also through a symbolic link)."""
    if os.path.isabs(path) or path.startswith("~"):
        raise PatchError(path + ": absolute paths are not allowed; use a path relative to " + root)
    full = os.path.realpath(os.path.join(root, path))
    base = os.path.realpath(root)
    if full == base or not full.startswith(base.rstrip(os.sep) + os.sep):
        raise PatchError(path + ": the path leaves the current directory")
    return full


def _find(lines, wanted, start, strip):
    """The first index >= start where ``wanted`` matches, or -1."""
    n = len(wanted)
    if strip:
        wanted = [w.rstrip() for w in wanted]
    for i in range(start, len(lines) - n + 1):
        ok = True
        for j in range(n):
            have = lines[i + j].rstrip() if strip else lines[i + j]
            if have != wanted[j]:
                ok = False
                break
        if ok:
            return i
    return -1


def _locate(lines, wanted, start):
    i = _find(lines, wanted, start, False)
    if i < 0:
        i = _find(lines, wanted, start, True)
    return i


def _first_mismatch(lines, wanted, start):
    """The first line of ``wanted`` after its longest prefix that is found in ``lines``."""
    for k in range(len(wanted) - 1, 0, -1):
        if _locate(lines, wanted[:k], start) >= 0:
            return wanted[k]
    return wanted[0]


def _split(content):
    """Lines without their ends, whether the file ends with a newline, and its line end."""
    eol = "\r\n" if "\r\n" in content else "\n"
    if eol == "\r\n":
        content = content.replace("\r\n", "\n")
    if content == "":
        return [], True, eol
    final = content.endswith("\n")
    if final:
        content = content[:-1]
    return content.split("\n"), final, eol


def _join(lines, final, eol):
    if not lines:
        return ""
    return eol.join(lines) + (eol if final else "")


def update(content, op):
    """``content`` with the hunks of ``op`` applied. Raises PatchError."""
    lines, final, eol = _split(content)
    cursor = 0
    for number, hunk in enumerate(op.hunks, 1):
        for header in hunk.headers:
            i = _locate(lines, [header], cursor)
            if i < 0:
                i = _find([x.strip() for x in lines], [header.strip()], cursor, False)
            if i < 0:
                raise PatchError(
                    "{}: hunk {} did not match: line not found: {!r}".format(
                        op.path, number, header
                    )
                )
            cursor = i + 1
        old = hunk.old
        if not old:
            # Only added lines: after the context line, else at the end of the file.
            at = len(lines) if (hunk.at_eof or not hunk.headers) else cursor
        else:
            at = -1
            if hunk.at_eof:
                end = len(lines) - len(old)
                if end >= cursor and _locate(lines, old, end) == end:
                    at = end
            if at < 0:
                at = _locate(lines, old, cursor)
            if at < 0:
                raise PatchError(
                    "{}: hunk {} did not match: line not found: {!r}".format(
                        op.path, number, _first_mismatch(lines, old, cursor)
                    )
                )
        new = hunk.replacement(lines[at : at + len(old)])
        lines[at : at + len(old)] = new
        cursor = at + len(new)
    return _join(lines, final, eol)


def _read(path):
    with open(path, encoding="utf-8", errors="surrogateescape", newline="") as f:
        return f.read()


def _write(path, content):
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(path, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
        f.write(content)


def apply(text, root):
    """Apply the patch ``text`` under ``root``; returns the summary lines. Nothing is written
    unless every section applies. Raises PatchError."""
    ops = parse(text)
    state = {}  # absolute path -> new content, or None for a deleted file
    summary = []
    for op in ops:
        try:
            summary.append(_plan(op, root, state))
        except OSError as exc:
            raise PatchError(op.path + ": " + str(exc)) from None
    # Deletions first, so a file can be deleted and its name used again.
    for full in sorted(state, key=lambda p: state[p] is not None):
        if state[full] is None:
            if os.path.isfile(full):
                os.remove(full)
        else:
            _write(full, state[full])
    return summary


def _exists(state, full):
    if full in state:
        return state[full] is not None
    return os.path.isfile(full)


def _plan(op, root, state):
    """Record the result of one file section in ``state`` (nothing is written here) and return
    its summary line."""
    full = check_path(root, op.path)
    if op.kind == "add":
        if _exists(state, full) or os.path.isdir(full):
            raise PatchError(op.path + ": cannot add, the file already exists (use an update)")
        state[full] = _join(op.lines, True, "\n")
        return "A " + op.path
    if op.kind == "delete":
        if not _exists(state, full):
            raise PatchError(op.path + ": cannot delete, no such file")
        state[full] = None
        return "D " + op.path
    if not _exists(state, full):
        raise PatchError(op.path + ": cannot update, no such file")
    new = update(state[full] if full in state else _read(full), op)
    if op.move_to is None or op.move_to == op.path:
        state[full] = new
        return "M " + op.path
    dest = check_path(root, op.move_to)
    if _exists(state, dest) or os.path.isdir(dest):
        raise PatchError(op.move_to + ": cannot move there, the file already exists")
    state[full] = None
    state[dest] = new
    return "M " + op.path + " -> " + op.move_to


def main(argv=None, stdin=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1:
        sys.stderr.write("usage: apply_patch [PATCH]   (or the patch on standard input)\n")
        return 2
    if argv:
        text = argv[0]
    else:
        stream = sys.stdin if stdin is None else stdin
        text = stream.read() if stream is not None else ""
    if not text.strip():
        sys.stderr.write(
            "apply_patch: no patch given (pass it on standard input or as one argument)\n"
        )
        return 2
    try:
        summary = apply(text, os.getcwd())
    except PatchError as exc:
        sys.stderr.write("apply_patch: " + str(exc) + "\nNo files were changed.\n")
        return 1
    except OSError as exc:
        sys.stderr.write("apply_patch: " + str(exc) + "\nThe patch may be partly applied.\n")
        return 1
    sys.stdout.write("Success. Updated the following files:\n" + "\n".join(summary) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
