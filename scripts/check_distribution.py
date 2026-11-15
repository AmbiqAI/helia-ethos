"""Verify built distributions through isolated installations outside the checkout."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def main() -> None:
    """Build wheel/sdist and exercise core and full-extra installed consumers."""
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    with tempfile.TemporaryDirectory(prefix="helia-ethos-distribution-") as directory:
        work = Path(directory)

        def run(*args: str, cwd: Path = work) -> None:
            subprocess.run(args, cwd=cwd, env=env, check=True)

        artifacts = work / "artifacts"
        run("uv", "build", "--out-dir", str(artifacts), cwd=root)
        wheels = list(artifacts.glob("*.whl"))
        sdists = list(artifacts.glob("*.tar.gz"))
        assert len(wheels) == len(sdists) == 1
        wheel = wheels[0]
        constraints = work / "constraints.txt"
        run("uv", "export", "--frozen", "--all-extras", "--no-dev", "--no-emit-project",
            "--no-hashes", "--output-file", str(constraints), cwd=root)
        venv = work / "venv"
        run("uv", "venv", "--python", sys.executable, str(venv))
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run("uv", "pip", "install", "--python", str(python), "--constraint", str(constraints), str(wheel))
        run(str(python), "-I", "-c", '''
from importlib.util import find_spec
from pathlib import Path
import helia_ethos
assert "site-packages" in str(Path(helia_ethos.__file__).resolve())
assert find_spec("helia_estimator") is None
assert find_spec("ai_edge_litert") is None
assert helia_ethos.load_soc("atomiq110").id == "atomiq110"
assert helia_ethos.load_target("atomiq110-fvp-corstone-320")
from helia_ethos.vela import write_system_config
write_system_config(helia_ethos.load_soc("atomiq110"), "ULP", Path("core.ini"))
''')
        run("uv", "pip", "install", "--python", str(python), "--constraint", str(constraints),
            f"{wheel}[all]", "pytest")
        shutil.copytree(root / "tests", work / "tests")
        shutil.copytree(root / "docs", work / "docs")
        # The installed entry point must be discoverable by the compile adapter.
        env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
        run(str(python), "-m", "pytest", "-q", str(work / "tests"))
        source = work / "sdist"
        with tarfile.open(sdists[0]) as archive:
            archive.extractall(source, filter="data")
        source_root, = source.iterdir()
        run("uv", "build", "--wheel", "--out-dir", str(work / "rebuilt"), cwd=source_root)
        # Install the sdist-built wheel too: package data must survive both routes.
        rebuilt, = (work / "rebuilt").glob("*.whl")
        run("uv", "pip", "install", "--python", str(python), "--reinstall-package", "helia-ethos",
            "--no-deps", str(rebuilt))
        run(str(python), "-m", "pytest", "-q", str(work / "tests/test_vela_compile.py"),
            str(work / "tests/test_catalog_policy.py"))
        output = root / "dist"
        output.mkdir(exist_ok=True)
        for artifact in (wheel, sdists[0]):
            shutil.copy2(artifact, output / artifact.name)
        print("Verified core/full-extra wheels, real Vela compilation and sdist rebuild.")


if __name__ == "__main__":
    main()
