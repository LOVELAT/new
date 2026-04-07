"""Baggage screening agent package."""


def initialize_agent(*args, **kwargs):
    from .main import initialize_agent as _initialize_agent

    return _initialize_agent(*args, **kwargs)


def create_demo(*args, **kwargs):
    from .interface import create_demo as _create_demo

    return _create_demo(*args, **kwargs)


__all__ = ["initialize_agent", "create_demo"]
