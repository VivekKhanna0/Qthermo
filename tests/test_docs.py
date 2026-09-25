"""Documentation hygiene: every public name is documented."""

import importlib
import inspect

import qthermo as qt


def test_every_public_function_and_class_has_a_docstring():
    missing = []
    for name in qt.__all__:
        obj = getattr(qt, name)
        if inspect.isfunction(obj) or inspect.isclass(obj):
            if not (inspect.getdoc(obj) or "").strip():
                missing.append(name)
    assert not missing, f"undocumented public names: {missing}"


def test_every_module_has_a_docstring():
    import pkgutil
    for info in pkgutil.iter_modules(qt.__path__):
        module = importlib.import_module(f"qthermo.{info.name}")
        assert (module.__doc__ or "").strip(), f"module {info.name} has no docstring"
