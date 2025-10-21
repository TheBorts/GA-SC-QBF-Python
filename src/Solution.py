# solutions/solution.py
from typing import Generic, TypeVar, List, Iterable

E = TypeVar("E")

class Solution(list):
    """A simple Python version of the Java Solution<E> class.
    It behaves like a list of elements and has a 'cost' attribute.
    """
    def __init__(self, iterable: Iterable = None):
        super().__init__(iterable if iterable is not None else [])
        self.cost = float("inf")

    def copy(self):
        s = Solution(self)
        s.cost = self.cost
        return s

    def __str__(self):
        return f"Solution: cost=[{self.cost}], size=[{len(self)}], elements={list.__str__(self)}"
