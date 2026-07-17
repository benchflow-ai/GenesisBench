from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import json
import mmap
import os
import pwd
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

ISOLATION_ENV = "GENESISBENCH_POLICY_ISOLATION"
RESTRICT_EXEC = Path("/usr/local/bin/restrict-exec")
WORKER_SCRIPT = Path("/opt/genesisbench/genesisbench/policy_isolation.py")
MAX_OBSERVATION_BYTES = 4 * 1024 * 1024
DEFAULT_CALL_TIMEOUT_SEC = 30.0


class PolicyIsolationError(RuntimeError):
    """Raised when a submitted policy cannot be executed safely."""


@dataclass(frozen=True)
class IsolatedPolicyModule:
    path: Path


def _isolation_mode() -> str:
    value = os.environ.get(ISOLATION_ENV, "auto").strip().lower()
    if value not in {"auto", "off", "required"}:
        raise PolicyIsolationError(
            f"{ISOLATION_ENV} must be auto, off, or required; got {value!r}"
        )
    return value


def isolation_enabled() -> bool:
    mode = _isolation_mode()
    if mode == "off":
        return False
    available = (
        sys.platform.startswith("linux")
        and RESTRICT_EXEC.is_file()
        and WORKER_SCRIPT.is_file()
    )
    if mode == "required" and not available:
        raise PolicyIsolationError(
            "required policy isolation is unavailable: expected "
            f"{RESTRICT_EXEC} and {WORKER_SCRIPT}"
        )
    return available


def _local_load(
    policy_path: Path,
    *,
    module_name: str,
    add_parent_to_path: bool,
    suppress_bytecode: bool,
) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, policy_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import policy from {policy_path}")
    module = importlib.util.module_from_spec(spec)
    parent = str(policy_path.parent)
    added_to_path = add_parent_to_path and parent not in sys.path
    previous = sys.dont_write_bytecode
    if added_to_path:
        sys.path.insert(0, parent)
    if suppress_bytecode:
        sys.dont_write_bytecode = True
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        if suppress_bytecode:
            sys.dont_write_bytecode = previous
        if added_to_path:
            sys.path.remove(parent)
    return module


def load_policy_module(
    policy_path: Path,
    *,
    module_name: str,
    add_parent_to_path: bool = False,
    suppress_bytecode: bool = False,
) -> ModuleType | IsolatedPolicyModule:
    path = policy_path.resolve()
    if isolation_enabled():
        return IsolatedPolicyModule(path)
    return _local_load(
        path,
        module_name=module_name,
        add_parent_to_path=add_parent_to_path,
        suppress_bytecode=suppress_bytecode,
    )


def _call_with_supported_kwargs(callable_: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError):
        return callable_(**kwargs)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return callable_(**kwargs)
    supported = {
        name: value for name, value in kwargs.items() if name in signature.parameters
    }
    return callable_(**supported)


def instantiate_policy(
    module: ModuleType | IsolatedPolicyModule,
    *,
    init_kwargs: dict[str, Any],
    missing_message: str = "Submission must define Policy or make_policy",
) -> Any:
    if isinstance(module, IsolatedPolicyModule):
        isolated_kwargs = dict(init_kwargs)
        if "seed" in isolated_kwargs:
            isolated_kwargs["seed"] = 0
        return IsolatedPolicy(
            policy_path=module.path,
            init_kwargs=isolated_kwargs,
        )
    if hasattr(module, "make_policy"):
        return _call_with_supported_kwargs(module.make_policy, init_kwargs)
    if hasattr(module, "Policy"):
        return _call_with_supported_kwargs(module.Policy, init_kwargs)
    raise AttributeError(missing_message)


def close_policy(policy: Any) -> None:
    close = getattr(policy, "close", None)
    if close is not None:
        close()


def _reject_unsafe_bundle(root: Path) -> None:
    file_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise PolicyIsolationError(f"policy bundle contains symlink: {path}")
        if not (stat.S_ISREG(metadata.st_mode) or stat.S_ISDIR(metadata.st_mode)):
            raise PolicyIsolationError(
                f"policy bundle contains special file: {path}"
            )
        if path.is_file():
            file_count += 1
            total_bytes += metadata.st_size
            if path.suffix in {".pth", ".pyc", ".so"}:
                raise PolicyIsolationError(
                    f"policy bundle contains forbidden file type: {path.name}"
                )
    if file_count > 2048:
        raise PolicyIsolationError("policy bundle contains too many files")
    if total_bytes > 512 * 1024 * 1024:
        raise PolicyIsolationError("policy bundle exceeds 512 MiB")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


class IsolatedPolicy:
    def __init__(
        self,
        *,
        policy_path: Path,
        init_kwargs: dict[str, Any],
        call_timeout_sec: float = DEFAULT_CALL_TIMEOUT_SEC,
    ) -> None:
        self.policy_path = policy_path.resolve()
        self.init_kwargs = _json_safe(init_kwargs)
        self.call_timeout_sec = float(call_timeout_sec)
        self._queued_commands: list[dict[str, Any]] = []
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._process: subprocess.Popen[str] | None = None
        self._observation_file = None
        self._observation_map: mmap.mmap | None = None
        self._worker_policy_path: Path | None = None
        self._stderr_lines: list[str] = []
        self._stderr_thread: threading.Thread | None = None

    def configure_simulator(self, **kwargs: Any) -> None:
        self._queued_commands.append(
            {"command": "configure", "kwargs": _json_safe(kwargs)}
        )

    def reset(self, seed: int = 0) -> None:
        del seed
        self._queued_commands.append(
            {"command": "reset", "kwargs": {"seed": 0}}
        )

    def _copy_bundle(self, destination: Path) -> Path:
        source_root = self.policy_path.parent
        _reject_unsafe_bundle(source_root)
        bundle = destination / "bundle"
        shutil.copytree(source_root, bundle)
        relative_policy = self.policy_path.relative_to(source_root)
        return bundle / relative_policy

    def _worker_command(self, root: Path, observation_path: Path) -> list[str]:
        try:
            account = pwd.getpwnam("agent")
        except KeyError as error:
            raise PolicyIsolationError("task image has no agent user") from error
        assert self._worker_policy_path is not None
        return [
            str(RESTRICT_EXEC),
            str(root),
            "--",
            "/usr/bin/setpriv",
            f"--reuid={account.pw_uid}",
            f"--regid={account.pw_gid}",
            "--clear-groups",
            "--no-new-privs",
            "--",
            sys.executable,
            "-I",
            str(WORKER_SCRIPT),
            "--worker",
            "--policy",
            str(self._worker_policy_path),
            "--observation",
            str(observation_path),
            "--init-json",
            json.dumps(self.init_kwargs, separators=(",", ":"), sort_keys=True),
        ]

    def _drain_stderr(self) -> None:
        assert self._process is not None
        assert self._process.stderr is not None
        for line in self._process.stderr:
            self._stderr_lines.append(line.rstrip())
            if len(self._stderr_lines) > 200:
                del self._stderr_lines[:50]

    def _start(self, observation: np.ndarray) -> None:
        if self._process is not None:
            return
        array = np.ascontiguousarray(observation)
        if array.nbytes > MAX_OBSERVATION_BYTES:
            raise PolicyIsolationError(
                f"observation requires {array.nbytes} bytes; "
                f"limit is {MAX_OBSERVATION_BYTES}"
            )
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="genesisbench-policy-"
        )
        root = Path(self._temporary_directory.name)
        self._worker_policy_path = self._copy_bundle(root)
        observation_path = root / "observation.bin"
        observation_path.write_bytes(b"\0" * MAX_OBSERVATION_BYTES)

        account = pwd.getpwnam("agent")
        for path in [root, *root.rglob("*")]:
            os.chown(path, account.pw_uid, account.pw_gid)
        os.chmod(root, 0o700)

        self._observation_file = observation_path.open("r+b")
        self._observation_map = mmap.mmap(
            self._observation_file.fileno(),
            MAX_OBSERVATION_BYTES,
        )
        environment = {
            "HOME": str(root),
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        self._process = subprocess.Popen(
            self._worker_command(root, observation_path),
            cwd=root,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            daemon=True,
        )
        self._stderr_thread.start()
        response = self._read_response()
        if not response.get("ok"):
            raise PolicyIsolationError(
                f"policy worker failed to initialize: {response.get('error')}"
            )
        queued = self._queued_commands
        self._queued_commands = []
        for command in queued:
            if command["command"] == "configure":
                kwargs = dict(command["kwargs"])
                model_path = kwargs.get("model_xml_path")
                if isinstance(model_path, str):
                    source = Path(model_path)
                    copied = root / "model.xml"
                    shutil.copy2(source, copied)
                    os.chown(copied, account.pw_uid, account.pw_gid)
                    kwargs["model_xml_path"] = str(copied)
                self._request({"command": "configure", "kwargs": kwargs})
            else:
                self._request(command)

    def _read_response(self) -> dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise PolicyIsolationError("policy worker is not running")
        if self._process.poll() is not None:
            detail = "\n".join(self._stderr_lines[-20:])
            raise PolicyIsolationError(
                f"policy worker exited with {self._process.returncode}: {detail}"
            )

        readable, _, _ = select.select(
            [self._process.stdout],
            [],
            [],
            self.call_timeout_sec,
        )
        if not readable:
            self.close()
            raise PolicyIsolationError(
                f"policy call exceeded {self.call_timeout_sec:.1f}s"
            )
        line = self._process.stdout.readline()
        if not line:
            detail = "\n".join(self._stderr_lines[-20:])
            raise PolicyIsolationError(
                f"policy worker closed its output stream: {detail}"
            )
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PolicyIsolationError(
                f"policy worker returned invalid JSON: {line[:200]!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise PolicyIsolationError("policy worker response must be an object")
        return payload

    def _request(self, payload: dict[str, Any]) -> Any:
        if self._process is None or self._process.stdin is None:
            raise PolicyIsolationError("policy worker is not running")
        self._process.stdin.write(
            json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
        )
        self._process.stdin.flush()
        response = self._read_response()
        if not response.get("ok"):
            raise PolicyIsolationError(str(response.get("error", "policy failed")))
        return response.get("result")

    def act(self, observation: Any, *args: Any, **kwargs: Any) -> Any:
        array = np.ascontiguousarray(observation)
        self._start(array)
        assert self._observation_map is not None
        self._observation_map.seek(0)
        self._observation_map.write(array.tobytes(order="C"))
        payload = {
            "command": "act",
            "observation": {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "nbytes": array.nbytes,
            },
            "args": _json_safe(args),
            "kwargs": _json_safe(kwargs),
        }
        return self._request(payload)

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is not None:
            if process.stdin is not None:
                try:
                    process.stdin.write('{"command":"close"}\n')
                    process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        if self._observation_map is not None:
            self._observation_map.close()
            self._observation_map = None
        if self._observation_file is not None:
            self._observation_file.close()
            self._observation_file = None
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
            self._temporary_directory = None

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass


def _worker_load_policy(policy_path: Path, init_kwargs: dict[str, Any]) -> Any:
    module_name = f"genesisbench_isolated_{abs(hash(policy_path))}"
    with contextlib.redirect_stdout(sys.stderr):
        module = _local_load(
            policy_path,
            module_name=module_name,
            add_parent_to_path=True,
            suppress_bytecode=True,
        )
        if hasattr(module, "make_policy"):
            return _call_with_supported_kwargs(module.make_policy, init_kwargs)
        if hasattr(module, "Policy"):
            return _call_with_supported_kwargs(module.Policy, init_kwargs)
    raise AttributeError("Submission must define Policy or make_policy")


def _worker_call_with_kwargs(target: Any, kwargs: dict[str, Any]) -> Any:
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return target(**kwargs)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return target(**kwargs)
    return target(
        **{name: value for name, value in kwargs.items() if name in signature.parameters}
    )


def _worker_main(args: argparse.Namespace) -> int:
    policy_path = args.policy.resolve()
    init_kwargs = json.loads(args.init_json)
    policy = _worker_load_policy(policy_path, init_kwargs)
    observation_file = args.observation.open("rb")
    observation_map = mmap.mmap(
        observation_file.fileno(),
        MAX_OBSERVATION_BYTES,
        access=mmap.ACCESS_READ,
    )
    print('{"ok":true}', flush=True)
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                command = request.get("command")
                if command == "close":
                    print('{"ok":true}', flush=True)
                    return 0
                if command == "configure":
                    configure = getattr(policy, "configure_simulator", None)
                    with contextlib.redirect_stdout(sys.stderr):
                        result = (
                            None
                            if configure is None
                            else _worker_call_with_kwargs(
                                configure,
                                dict(request.get("kwargs") or {}),
                            )
                        )
                elif command == "reset":
                    reset = getattr(policy, "reset", None)
                    with contextlib.redirect_stdout(sys.stderr):
                        result = (
                            None
                            if reset is None
                            else _worker_call_with_kwargs(
                                reset,
                                dict(request.get("kwargs") or {}),
                            )
                        )
                elif command == "act":
                    metadata = request["observation"]
                    nbytes = int(metadata["nbytes"])
                    dtype = np.dtype(metadata["dtype"])
                    shape = tuple(int(item) for item in metadata["shape"])
                    expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                    if nbytes != expected or nbytes > MAX_OBSERVATION_BYTES:
                        raise ValueError("invalid observation metadata")
                    observation = np.ndarray(
                        shape,
                        dtype=dtype,
                        buffer=observation_map,
                    ).copy()
                    with contextlib.redirect_stdout(sys.stderr):
                        result = policy.act(
                            observation,
                            *(request.get("args") or []),
                            **(request.get("kwargs") or {}),
                        )
                else:
                    raise ValueError(f"unknown worker command: {command!r}")
                print(
                    json.dumps(
                        {"ok": True, "result": _json_safe(result)},
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except BaseException as exc:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        observation_map.close()
        observation_file.close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--init-json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.worker:
        raise SystemExit("policy_isolation.py is an internal worker")
    if args.policy is None or args.observation is None or args.init_json is None:
        raise SystemExit("worker arguments are incomplete")
    return _worker_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
