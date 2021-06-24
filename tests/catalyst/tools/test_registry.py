# flake8: noqa
import pytest

from catalyst.tools.registry import Registry, RegistryException

from . import registery_foo as module
from .registery_foo import foo


def test_add_function():
    """@TODO: Docs. Contribution is welcome."""
    r = Registry()

    r.add(foo)

    assert "foo" in r._factories  # noqa: WPS437


def test_add_function_name_override():
    """@TODO: Docs. Contribution is welcome."""
    r = Registry()

    r.add(foo, name="bar")

    assert "bar" in r._factories  # noqa: WPS437


def test_add_lambda_fail():
    """@TODO: Docs. Contribution is welcome."""
    r = Registry()

    with pytest.raises(RegistryException):
        r.add(lambda x: x)


def test_add_lambda_override():
    """@TODO: Docs. Contribution is welcome."""
    r = Registry()

    r.add(lambda x: x, name="bar")

    assert "bar" in r._factories  # noqa: WPS437


def test_fail_multiple_with_name():
    """@TODO: Docs. Contribution is welcome."""
    r = Registry()

    with pytest.raises(RegistryException):
        r.add(foo, foo, name="bar")


def test_fail_double_add_different():
    """@TODO: Docs. Contribution is welcome."""
    r = Registry()
    r.add(foo)

    with pytest.raises(RegistryException):

        def bar():
            pass

        r.add(foo=bar)


def test_double_add_same_nofail():
    """@TODO: Docs. Contribution is welcome."""
    r = Registry()
    r.add(foo)
    # It's ok to add same twice, forced by python relative import
    # implementation
    # https://github.com/catalyst-team/catalyst/issues/135
    r.add(foo)


def test_instantiations():
    """@TODO: Docs. Contribution is welcome."""
    r = Registry()

    r.add(foo)

    res = r.get_instance("foo", 1, 2)()
    assert res == {"a": 1, "b": 2}

    res = r.get_instance("foo", 1, b=2)()
    assert res == {"a": 1, "b": 2}

    res = r.get_instance("foo", a=1, b=2)()
    assert res == {"a": 1, "b": 2}

    res = r.get_instance("tests.catalyst.tools.registery_foo.foo", a=1, b=2)()
    assert res == {"a": 1, "b": 2}


def test_from_config():
    """@TODO: Docs. Contribution is welcome."""
    r = Registry()

    r.add(foo)

    res = r.get_from_params(**{"_target_": "foo", "a": 1, "b": 2})()
    assert res == {"a": 1, "b": 2}

    res = r.get_from_params(**{})
    assert res == {}


def test_name_key():
    """@TODO: Docs. Contribution is welcome."""
    r = Registry(name_key="_key_")

    r.add(foo)

    res = r.get_from_params(**{"_key_": "foo", "a": 1, "b": 2})()
    assert res == {"a": 1, "b": 2}

    res = r.get_from_params(**{"_target_": "foo", "a": 1, "b": 2})
    assert res == {"_target_": "foo", "a": 1, "b": 2}


def test_recursive_get_from_config():
    def meta_factory(factory, args, kwargs):
        return factory(*args, **kwargs)

    r = Registry(default_meta_factory=meta_factory)

    r.add(foo)

    res = r.get_from_params(**{"_target_": "foo", "a": 1, "b": 2})
    assert res == {"a": 1, "b": 2}

    res = r.get_from_params(
        **{
            "_target_": "foo",
            "a": {"_target_": "foo", "a": {"_target_": "foo", "a": 1, "b": 2}, "b": 2},
            "b": [{"_target_": "foo", "a": 1, "b": 2}, {"_target_": "foo", "a": 1, "b": 2}],
        }
    )
    assert res == {"a": {"a": {"a": 1, "b": 2}, "b": 2}, "b": [{"a": 1, "b": 2}, {"a": 1, "b": 2}]}

    res = r.get_from_params(
        **{"a": {"_target_": "foo", "a": 1, "b": 2}, "b": {"_target_": "foo", "a": 1, "b": 2}}
    )
    assert res == {"a": {"a": 1, "b": 2}, "b": {"a": 1, "b": 2}}

    # check nested dicts support
    res = r.get_from_params(
        **{"_target_": "foo", "a": {"c": {"_target_": "foo", "a": 1, "b": 2}}, "b": 2}
    )
    assert res == {"a": {"c": {"a": 1, "b": 2}}, "b": 2}

    # check nested lists support
    res = r.get_from_params(
        **{
            "_target_": "foo",
            "a": [[[{"_target_": "foo", "a": 1, "b": 2}]]],
            "b": {"c": 3, "d": {"e": 4}},
        }
    )
    assert res == {"a": [[[{"a": 1, "b": 2}]]], "b": {"c": 3, "d": {"e": 4}}}

    # check shared_params
    res = r.get_from_params(
        **{"_target_": "foo", "a": {"_target_": "foo", "a": 1, "b": 3}}, shared_params={"b": 2}
    )
    assert res == {"a": {"a": 1, "b": 3}, "b": 2}


def test_meta_factory():
    """@TODO: Docs. Contribution is welcome."""  # noqa: D202

    def meta_factory1(fn, args, kwargs):
        return fn

    def meta_factory2(fn, args, kwargs):
        return 1

    r = Registry(meta_factory1)
    r.add(foo)

    res = r.get_from_params(**{"_target_": "foo"})
    assert res == foo

    res = r.get_from_params(**{"_target_": "foo"}, meta_factory=meta_factory2)
    assert res == 1


def test_fail_instantiation():
    """@TODO: Docs. Contribution is welcome."""
    r = Registry()

    r.add(foo)

    with pytest.raises((RegistryException, TypeError)) as e_ifo:
        r.get_instance("foo", c=1)()

    assert hasattr(e_ifo.value, "__cause__")


def test_decorator():
    """@TODO: Docs. Contribution is welcome."""
    r = Registry()

    @r.add
    def bar():
        pass

    r.get("bar")


def test_kwargs():
    """@TODO: Docs. Contribution is welcome."""
    r = Registry()

    r.add(bar=foo)

    r.get("bar")


def test_add_module():
    """@TODO: Docs. Contribution is welcome."""
    r = Registry()

    r.add_from_module(module)

    r.get("foo")

    with pytest.raises(RegistryException):
        r.get_instance("bar")
