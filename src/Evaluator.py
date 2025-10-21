from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from Solution import Solution

E = TypeVar("E")

class Evaluator(ABC, Generic[E]):
    """
    The Evaluator interface (abstract base class in Python).
    Provides methods to evaluate solutions and local cost changes.
    """

    @abstractmethod
    def getDomainSize(self) -> int:
        """Gives the size of the problem domain."""
        pass

    @abstractmethod
    def evaluate(self, sol: Solution) -> float:
        """Returns the evaluation (cost or fitness) of a solution."""
        pass

    @abstractmethod
    def evaluateInsertionCost(self, elem: E, sol: Solution) -> float:
        """Evaluates the cost variation of inserting an element into a solution."""
        pass

    @abstractmethod
    def evaluateRemovalCost(self, elem: E, sol: Solution) -> float:
        """Evaluates the cost variation of removing an element from a solution."""
        pass

    @abstractmethod
    def evaluateExchangeCost(self, elemIn: E, elemOut: E, sol: Solution) -> float:
        """Evaluates the cost variation of exchanging one element for another."""
        pass
