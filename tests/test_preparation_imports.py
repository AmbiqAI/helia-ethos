"""Importing public types and entry points must not load compiler implementations."""

import subprocess
import sys
import textwrap


def test_public_preparation_imports_do_not_load_compiler_or_optional_dependencies():
    """Offline consumers can import public names even when compiler imports are blocked."""
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        blocked = ('helia_ethos.vela', 'helia_ethos.preparation.prepare',
                   'ethosu', 'ai_edge_litert')
        class OfflineImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
                    raise AssertionError(f'offline import loaded {fullname}')
        sys.meta_path.insert(0, OfflineImports())
        import helia_ethos
        from helia_ethos import PrepareRequest, prepare_model, inspect_prepared_model
        from helia_ethos.preparation import PreparationRecord
        from helia_ethos.pmu import PmuProfile
        assert callable(prepare_model) and callable(inspect_prepared_model)
        assert PrepareRequest and PreparationRecord and PmuProfile
        for name in helia_ethos.__all__:
            getattr(helia_ethos, name)
    """)
    completed = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def test_prepared_model_runtime_type_hints_resolve():
    """Opt-in runtime introspection resolves the same compiler result type."""
    from typing import get_type_hints

    from helia_ethos import PreparedModel
    from helia_ethos.vela import CompiledModel

    assert get_type_hints(PreparedModel)["compilation"] is CompiledModel
