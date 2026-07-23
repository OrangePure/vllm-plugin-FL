# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from https://github.com/vllm-project/vllm-ascend/blob/main/vllm_ascend/worker/worker.py
# Below is the original copyright:
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Ascend static-kernel lifecycle patch (vllm-ascend ``enable_static_kernel`` port).

When ``enable_static_kernel`` is on, npugraph_ex/torchair pre-compiles operator
binaries with fixed shapes during aclgraph capture (see
``..compilation.compiler_interface``) and installs them as .run packages under
``$ASCEND_HOME_PATH/opp/static_kernel`` — process-external global state that
must be uninstalled when the worker exits.

This patch is the ONLY generic-code-free hook needed for that lifecycle:
``patch_static_kernel()`` runs inside ``apply_ascend_patches()`` during
``WorkerFL.__init__`` (via ``register_oot_ops``) and, when the feature is
enabled, registers SIGTERM/SIGINT handlers that uninstall the static kernel
packages exactly once (fcntl file lock across worker processes).

The feature is disabled by default.  Enable it the same way as vllm-ascend:

* ``--additional-config '{"ascend_compilation_config":
  {"enable_npugraph_ex": true, "enable_static_kernel": true}}'``, or
* env ``VLLM_FL_ENABLE_NPUGRAPH_EX=1`` + ``VLLM_FL_ENABLE_STATIC_KERNEL=1``.

It requires npugraph_ex (the compile backend carrying the static-kernel
options) and is not supported on 310P.
"""

import logging
import os

logger = logging.getLogger(__name__)

_patched = False


def is_static_kernel_enabled() -> bool:
    """Whether static-kernel precompilation is enabled and supported.

    Mirrors vllm-ascend's config validation (``ascend_config.py``): requires
    ``enable_npugraph_ex`` and is force-disabled on 310P.  Both flags accept
    either the env var or ``additional_config["ascend_compilation_config"]``.
    """
    enabled = os.environ.get("VLLM_FL_ENABLE_STATIC_KERNEL", "0") == "1"
    npugraph_ex = os.environ.get("VLLM_FL_ENABLE_NPUGRAPH_EX", "0") == "1"
    if not (enabled and npugraph_ex):
        try:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
            if vllm_config is not None:
                ascend_cfg = (vllm_config.additional_config or {}).get(
                    "ascend_compilation_config", {}
                )
                enabled = enabled or ascend_cfg.get("enable_static_kernel", False)
                npugraph_ex = npugraph_ex or ascend_cfg.get("enable_npugraph_ex", False)
        except Exception:
            pass
    if enabled and not npugraph_ex:
        logger.warning(
            "enable_static_kernel requires enable_npugraph_ex to be enabled. "
            "Ignoring enable_static_kernel."
        )
        return False
    if not (enabled and npugraph_ex):
        return False
    try:
        from vllm_ascend.utils import is_310p

        if is_310p():
            logger.warning(
                "static kernel requires npugraph_ex, which is not supported on "
                "Ascend 310P. Ignoring enable_static_kernel."
            )
            return False
    except ImportError:
        pass
    return True


def uninstall_static_kernel() -> None:
    """Uninstall the static-kernel .run packages (vllm-ascend port).

    Uses a non-blocking fcntl lock so that among all worker processes on the
    node only one runs ``opp/static_kernel/ai_core/uninstall.sh``.
    """
    import fcntl
    import subprocess

    ascend_home_path = os.environ["ASCEND_HOME_PATH"]
    static_kernel_dir_path = os.path.join(ascend_home_path, "opp/static_kernel")
    uninstall_script_path = os.path.join(static_kernel_dir_path, "ai_core/uninstall.sh")
    lock_file_path = os.path.join(static_kernel_dir_path, "uninstall.lock")

    if not os.path.exists(uninstall_script_path):
        return
    with open(lock_file_path, "w") as lock_fd:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            subprocess.Popen(
                ["bash", uninstall_script_path],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (BlockingIOError, OSError):
            return
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                if os.path.exists(lock_file_path):
                    os.remove(lock_file_path)
            except Exception:
                return


def patch_static_kernel() -> bool:
    """Register the static-kernel uninstall hook when the feature is enabled.

    Idempotent.  Called from ``apply_ascend_patches()`` during worker init;
    the signal handlers therefore live in the worker process, same as
    vllm-ascend's NPUWorker hook.
    """
    global _patched
    if _patched:
        return False
    _patched = True

    if not is_static_kernel_enabled():
        return False

    # Prevent duplicate triggers, execute the exit logic only once
    shutdown_request = False

    def signal_handler(signum, frame):
        nonlocal shutdown_request
        if not shutdown_request:
            shutdown_request = True
            uninstall_static_kernel()
            raise SystemExit()

    # Either SIGTERM or SIGINT will terminate the worker
    import signal

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    logger.info(
        "enable_static_kernel is on: registered static-kernel uninstall hook "
        "(SIGTERM/SIGINT) for Ascend"
    )
    return True
