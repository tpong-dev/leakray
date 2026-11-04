"""Leakage rules. Each ``lkNNN_*.py`` module registers one rule."""

from .base import Rule, all_rules, get_rule, register

__all__ = ["Rule", "all_rules", "get_rule", "register"]
