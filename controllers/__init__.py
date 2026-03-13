"""
Balance controller interface and factory.

All balance controllers implement BalanceControllerBase so that the sim
loop can treat them uniformly — no hasattr() branching.
"""

from controllers.base import BalanceControllerBase

__all__ = ['BalanceControllerBase']
