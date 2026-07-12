"""Low-level helpers for the REAL OfficeBench benchmark (host side: Docker orchestration only).

OfficeBench solves office tasks by an agent that emits JSON actions ``{'app','action',...}`` against
a set of office apps, executed inside a Docker container whose ``/testbed`` filesystem is the task's
backing state, and scored by execution-based, file-state checks (``utils/evaluate.py``).  This module
provides the **host-side** pieces the trajectory wrapper needs:

* availability / data-root resolution (``OFFICEBENCH_ROOT`` -> the OfficeBench clone);
* the official subtask taxonomy enumeration (grouped by app-count: ``1app`` / ``2app`` / ``3app``);
* per-subtask config loading (plain JSON — username/date/task/evaluation);
* a per-rollout Docker **session** that mirrors the official runner: it materialises a fresh
  ``/testbed`` (+ ``/apps`` + ``/utils``) inside a throwaway container and drives every action / the
  evaluation through the in-container :mod:`tts.env.officebench_driver` (so all OfficeBench-specific
  Python deps live in the image, never on the host).

Nothing here imports OfficeBench's Python packages — the host needs only the ``docker`` CLI and the
``OFFICEBENCH_ROOT`` clone (for the task tree + the ``apps``/``utils`` directories it copies in).
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import threading
import uuid
from typing import Dict, List, Optional, Tuple

#: The image the apps + evaluation run in (built on demand from ``officebench_docker/``).
_DEFAULT_IMAGE = "officebench-tts:latest"
_DRIVER_PATH = os.path.join(os.path.dirname(__file__), "officebench_driver.py")
_DOCKER_DIR = os.path.join(os.path.dirname(__file__), "officebench_docker")
_RESULT_PREFIX = "OFFICEBENCH_RESULT="

_NAME_LOCK = threading.Lock()
_NAME_COUNTER = 0

#: ``"<apps>-<n>"`` task id (e.g. ``"3-42"``); the leading digit is the app-count.
_TASK_ID_RE = re.compile(r"^(\d+)-\d+$")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Root / availability
# --------------------------------------------------------------------------- #


def _default_root() -> Optional[str]:
    return os.environ.get("OFFICEBENCH_ROOT") or os.environ.get("OFFICEBENCH_DATA_ROOT")


def officebench_root(root: Optional[str] = None) -> Optional[str]:
    """Resolve the OfficeBench clone root, or ``None`` if it isn't a usable checkout."""
    root = root or _default_root()
    if not root:
        return None
    if (os.path.isdir(os.path.join(root, "tasks"))
            and os.path.isdir(os.path.join(root, "apps"))
            and os.path.isfile(os.path.join(root, "utils", "evaluate.py"))):
        return root
    return None


def docker_available() -> bool:
    """True if the ``docker`` CLI is usable (binary present + daemon reachable)."""
    try:
        return subprocess.run(["docker", "info"], capture_output=True, text=True,
                              timeout=30).returncode == 0
    except Exception:
        return False


def real_officebench_available(root: Optional[str] = None) -> bool:
    """OfficeBench is runnable iff the clone is present AND docker is usable.

    (The app/eval image is built on demand by :meth:`OfficeBenchSession.start`; use
    :func:`officebench_image_available` to gate tests that should not pay the one-off build.)
    """
    return officebench_root(root) is not None and docker_available()


def officebench_image() -> str:
    return os.environ.get("TTS_OFFICEBENCH_IMAGE", _DEFAULT_IMAGE)


def officebench_image_available() -> bool:
    try:
        return subprocess.run(["docker", "image", "inspect", officebench_image()],
                              capture_output=True, text=True, timeout=30).returncode == 0
    except Exception:
        return False


def ensure_image(build: bool = True) -> str:
    """Return the app/eval image name, building it from ``officebench_docker/`` if missing."""
    name = officebench_image()
    if officebench_image_available():
        return name
    if not build:
        raise RuntimeError(f"OfficeBench image {name!r} is not built; run a build first.")
    res = subprocess.run(["docker", "build", "-t", name, "."], cwd=_DOCKER_DIR,
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"failed to build OfficeBench image {name!r}:\n{res.stderr[-2000:]}")
    return name


# --------------------------------------------------------------------------- #
# Task taxonomy enumeration (host side, pure directory listing)
# --------------------------------------------------------------------------- #


def load_officebench_subtasks(app_count: int, root: Optional[str] = None) -> List[Tuple[str, str]]:
    """Sorted ``[(task_id, subtask_id), ...]`` for one app-count group (official ordering).

    Mirrors the official ``evaluation.py`` enumeration: every ``tasks/<id>/subtasks/<k>.json`` whose
    ``<id>`` starts with ``app_count`` is enumerated, sorted by ``(task numbers..., subtask number)``.
    """
    base = officebench_root(root)
    if base is None:
        raise RuntimeError("OfficeBench clone not found; export OFFICEBENCH_ROOT.")
    pairs: List[Tuple[str, str]] = []
    for path in glob.glob(os.path.join(base, "tasks", f"{app_count}-*", "subtasks", "*.json")):
        parts = path.split(os.sep)
        task_id = parts[-3]
        subtask_id = os.path.splitext(parts[-1])[0]
        m = _TASK_ID_RE.match(task_id)
        if m and int(m.group(1)) == app_count and subtask_id.isdigit():
            pairs.append((task_id, subtask_id))
    pairs.sort(key=lambda p: (tuple(int(x) for x in p[0].split("-")), int(p[1])))
    return pairs


def officebench_task_count(app_count: int, root: Optional[str] = None) -> int:
    return len(load_officebench_subtasks(app_count, root))


def load_subtask_config(task_id: str, subtask_id: str, root: Optional[str] = None) -> Dict:
    """Load one subtask's config JSON (username/date/weekday/time/task/evaluation)."""
    base = officebench_root(root)
    if base is None:
        raise RuntimeError("OfficeBench clone not found; export OFFICEBENCH_ROOT.")
    path = os.path.join(base, "tasks", task_id, "subtasks", f"{subtask_id}.json")
    with open(path) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Orphan-container sweeping (parent-side safety net for hard-killed workers)
# --------------------------------------------------------------------------- #

#: docker label stamped on every session container so a run can reap its own orphans.
_RUN_LABEL = "tts_officebench_run"
#: opt-out of the automatic container sweep (debugging).
_KEEP_CONTAINERS_ENV = "TTS_KEEP_OFFICEBENCH_CONTAINERS"


def _run_id() -> str:
    """This run's id (namespaces containers so a parent sweep can't touch another run's)."""
    return os.environ.get("TTS_OFFICEBENCH_RUN_ID", "default")


def sweep_officebench_containers(run_id: Optional[str] = None) -> int:
    """Force-remove every session container of ``run_id`` (default: this run's).

    A safety net for orphans left by **hard-killed** workers (SIGKILL on OOM/timeout skips the
    per-session ``finally: close()``), mirroring AppWorld's ``sweep_experiment_outputs``.  Call from
    the parent process at the end of a sweep with the run-unique id so it can never remove another
    concurrent run's containers.  Returns the number removed.
    """
    if os.environ.get(_KEEP_CONTAINERS_ENV):
        return 0
    rid = run_id or _run_id()
    try:
        listed = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"label={_RUN_LABEL}={rid}"],
            capture_output=True, text=True, timeout=60)
    except Exception:
        return 0
    ids = [x for x in (listed.stdout or "").split() if x]
    if not ids:
        return 0
    try:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, text=True, timeout=120)
    except Exception:
        return 0
    return len(ids)


# --------------------------------------------------------------------------- #
# Per-rollout Docker session
# --------------------------------------------------------------------------- #


def _unique_name(task_id: str, subtask_id: str, phase: str) -> str:
    global _NAME_COUNTER
    with _NAME_LOCK:
        _NAME_COUNTER += 1
        n = _NAME_COUNTER
    safe = f"{task_id}_{subtask_id}".replace("-", "_")
    return f"officebench_tts_{safe}_{phase}_p{os.getpid()}_{n}_{uuid.uuid4().hex[:8]}"


class OfficeBenchSession:
    """A throwaway Docker container holding ONE rollout's ``/testbed`` (the task's backing state).

    Faithful to :class:`utils.env.OfficeAgentEnv`: the per-task ``testbed/`` (input files) + the
    official ``apps`` and ``utils`` packages are copied into the container, every action and the
    evaluation are executed in-container through :mod:`tts.env.officebench_driver`, and the container
    is removed on :meth:`close`.  Each rollout gets its own session, so the real env holds no shared
    mutable state across rollouts and a world-model (shadow) rollout never touches another's files.
    """

    def __init__(self, task_id: str, subtask_id: str, *, phase: str = "run",
                 root: Optional[str] = None, image: Optional[str] = None,
                 exec_timeout_s: Optional[float] = None, read_only: bool = False) -> None:
        self.task_id = task_id
        self.subtask_id = subtask_id
        self.root = officebench_root(root)
        if self.root is None:
            raise RuntimeError("OfficeBench clone not found; export OFFICEBENCH_ROOT.")
        self.image = image or officebench_image()
        self.name = _unique_name(task_id, subtask_id, phase)
        # Heavy apps (LibreOffice excel/word->pdf, tesseract OCR, pdf2docx) can be slow, especially
        # under high --workers contention (many containers each spawning soffice).  The per-action
        # timeout is generous and tunable; on timeout the action degrades to an error observation
        # rather than killing the whole rollout (see _driver).
        if exec_timeout_s is None:
            exec_timeout_s = _env_float("TTS_OFFICEBENCH_EXEC_TIMEOUT_S", 300.0)
        self.exec_timeout_s = exec_timeout_s
        # When ``read_only`` (the world-model sampling container), executing a WRITE action is a bug:
        # the shadow must never be mutated.  ``exec_action`` refuses write actions structurally, so a
        # routing mistake fails loudly instead of silently writing state.
        self.read_only = read_only
        self._started = False
        self._apps_cache: Optional[Dict] = None

    # -- lifecycle ------------------------------------------------------- #
    def start(self) -> "OfficeBenchSession":
        ensure_image()
        self._docker("run", "-d", "--name", self.name, "--label", f"{_RUN_LABEL}={_run_id()}",
                     "--entrypoint", "", self.image, "sleep", "infinity")
        self._started = True
        # Lay out the container exactly like the official runner: /apps, /utils, /testbed, + driver.
        self._exec("mkdir", "-p", "/testbed/data", "/testbed/emails", "/testbed/calendar")
        self._cp(_DRIVER_PATH, f"{self.name}:/officebench_driver.py")
        self._cp(os.path.join(self.root, "apps"), f"{self.name}:/")
        self._cp(os.path.join(self.root, "utils"), f"{self.name}:/")
        testbed = os.path.join(self.root, "tasks", self.task_id, "testbed")
        if os.path.isdir(testbed):
            # copy the task's input files INTO the prepared /testbed (note the trailing '/.')
            self._cp(os.path.join(testbed, "."), f"{self.name}:/testbed")
        # Some tasks ship gold expected-output files under tasks/<id>/reference/ (used by the
        # evaluate_exact_match checks).  Copy them to /reference so the reference-grounded verifier
        # can read the gold artifact in-container with the official helpers (read-only).
        reference = os.path.join(self.root, "tasks", self.task_id, "reference")
        if os.path.isdir(reference):
            self._cp(reference, f"{self.name}:/reference")
        return self

    def close(self) -> None:
        if self._started:
            try:
                self._docker("rm", "-f", self.name)
            finally:
                self._started = False

    def __enter__(self) -> "OfficeBenchSession":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- driver-backed operations --------------------------------------- #
    def apps_catalogue(self) -> Dict:
        """The official app catalogue (INTRO + per-action DEMO), fetched once and cached."""
        if self._apps_cache is None:
            self._apps_cache = self._driver("apps", {})
        return self._apps_cache

    def exec_action(self, action: Dict, workdir: str) -> Dict:
        """Execute ONE app/shell action; returns ``{observation, exit_code, is_cd, new_workdir}``.

        A heavy/hung action (e.g. LibreOffice convert under load) degrades to an error observation
        rather than aborting the whole rollout, so a single timeout doesn't lose the trajectory.
        On a ``read_only`` session a WRITE action raises — the shadow must never be mutated — and any
        file an exact READ incidentally creates (create-if-missing read helpers) is reverted in-driver,
        so the read-only shadow stays byte-for-byte pristine.
        """
        if self.read_only:
            from .officebench_wm_predict import is_write_action

            if is_write_action(action):
                raise PermissionError(
                    f"write action {action.get('app')}.{action.get('action')} attempted on a read-only "
                    "world-model session (writes must be predicted, never executed)")
        res = self._driver("exec", {"action": action, "workdir": workdir,
                                    "read_only": self.read_only})
        if res.get("_timeout"):
            return {"observation": "OBSERVATION: action timed out and was skipped. Try a simpler step.",
                    "exit_code": 1, "is_cd": False, "new_workdir": None, "command": ""}
        return res

    def write_answer(self, answer: str) -> Dict:
        return self._driver("write_answer", {"answer": answer})

    def evaluate(self, evaluation: List[Dict]) -> Dict:
        """Run the task's ground-truth ``evaluation`` config against ``/testbed`` (the official score)."""
        res = self._driver("evaluate", {"evaluation": evaluation, "testbed_dir": "/testbed"})
        if res.get("_timeout"):
            return {"passed": False, "num_checks": 0, "num_passed": 0, "checks": [], "timeout": True}
        return res

    def read_state(self, files: List[Dict]) -> Dict:
        """Read the ACTUAL resulting ``/testbed`` state (read-only) for the grounded verifier.

        ``files`` is ``[{"path","doc_type"}, ...]`` (the result files the ground-truth checks
        reference, derived on the host from the eval config); the driver also adds ``answer.txt``,
        calendars and email accounts.  Never executes the candidate or the official boolean checks.
        """
        return self._driver("read_state", {"files": files, "extras": True}).get("state", {})

    def read_reference(self, files: List[Dict]) -> Dict:
        """Read the GOLD expected-output files (``/reference/...``) for the reference verifier.

        ``files`` is ``[{"path","doc_type"}, ...]`` pointing at the in-container ``/reference`` copies
        of ``tasks/<id>/reference/*``.  Read-only, no extras (only the requested gold files).
        """
        return self._driver("read_state", {"files": files, "extras": False}).get("state", {})

    def testbed_digest(self) -> str:
        """A content hash of the whole ``/testbed`` tree (to PROVE the WM shadow is never written).

        Excludes ``data/answer.txt`` (the agent's answer, written only on the committed path).  Two
        digests taken before and after a world-model rollout must be equal — a real write-safety check,
        unlike the trivially-constant :meth:`OfficeBenchTaskEnv.state_hash`.
        """
        script = (
            "cd /testbed && find . -type f ! -name answer.txt -printf '%p\\n' | LC_ALL=C sort | "
            "while read f; do md5sum \"$f\"; done | md5sum | cut -d' ' -f1"
        )
        res = self._docker("exec", self.name, "bash", "-c", script)
        return (res.stdout or "").strip()

    # -- low-level docker ----------------------------------------------- #
    def _driver(self, cmd: str, payload: Dict) -> Dict:
        try:
            proc = subprocess.run(
                ["docker", "exec", "-i", self.name, "python3", "/officebench_driver.py", cmd],
                input=json.dumps(payload), capture_output=True, text=True, timeout=self.exec_timeout_s,
            )
        except subprocess.TimeoutExpired:
            # Heavy app (LibreOffice/tesseract/pdf2docx) under high --workers contention: don't crash
            # the rollout — callers turn ``_timeout`` into a graceful failure observation.  ``apps`` is
            # the one fatal case (no catalogue -> no prompt), so it re-raises.
            if cmd == "apps":
                raise
            return {"_timeout": True}
        for line in reversed((proc.stdout or "").splitlines()):
            if line.startswith(_RESULT_PREFIX):
                return json.loads(line[len(_RESULT_PREFIX):])
        raise RuntimeError(
            f"officebench driver '{cmd}' returned no result (rc={proc.returncode}): "
            f"{(proc.stderr or proc.stdout)[-500:]}")

    def _exec(self, *args: str) -> subprocess.CompletedProcess:
        return self._docker("exec", self.name, *args)

    def _cp(self, src: str, dst: str) -> None:
        self._docker("cp", src, dst)

    def _docker(self, *args: str) -> subprocess.CompletedProcess:
        res = subprocess.run(["docker", *args], capture_output=True, text=True,
                             timeout=self.exec_timeout_s)
        if res.returncode != 0:
            raise RuntimeError(f"docker {' '.join(args[:2])} failed: {res.stderr.strip()[-500:]}")
        return res


__all__ = [
    "officebench_root",
    "docker_available",
    "real_officebench_available",
    "officebench_image",
    "officebench_image_available",
    "ensure_image",
    "load_officebench_subtasks",
    "officebench_task_count",
    "load_subtask_config",
    "sweep_officebench_containers",
    "OfficeBenchSession",
]
