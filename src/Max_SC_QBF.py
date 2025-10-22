from typing import List
from Solution import Solution
from Evaluator import Evaluator
import re

# Simple BitSet class to mimic Java's BitSet behavior
class BitSet:
    def __init__(self, size: int):
        self.size = size
        self._set = set()

    def set(self, idx: int):
        if 0 <= idx < self.size:
            self._set.add(idx)

    def or_(self, other: "BitSet"):
        self._set |= other._set

    def and_(self, other: "BitSet"):
        self._set &= other._set

    def flip(self, start: int, end: int):
        """Flips bits in the range [start, end)."""
        for i in range(start, end):
            if i in self._set:
                self._set.remove(i)
            else:
                self._set.add(i)

    def cardinality(self) -> int:
        return len(self._set)

    def nextSetBit(self, start: int) -> int:
        """Returns the smallest set bit >= start, or -1 if none."""
        candidates = [x for x in self._set if x >= start]
        return min(candidates) if candidates else -1

    def clone(self):
        b = BitSet(self.size)
        b._set = set(self._set)
        return b

    def get(self, idx: int) -> bool:
        return idx in self._set

    def __str__(self):
        return "{" + ", ".join(map(str, sorted(self._set))) + "}"


class MAX_SC_QBF(Evaluator):
    """Python version of the MAX_SC_QBF problem (Set Cover Quadratic Binary Function)."""

    def __init__(self, filename: str):
        self.size = self.readInput(filename)
        self.variables = self.allocateVariables()

    def setVariables(self, sol: Solution):
        self.resetVariables()
        for elem in sol:
            self.variables[elem] = 1.0

    def getDomainSize(self) -> int:
        return self.size

    def evaluate(self, sol: Solution) -> float:
        self.setVariables(sol)
        sol.cost = self.evaluateMAXSCQBF()
        return sol.cost

    def getValue(self, elem: int) -> float:
        return sum(self.A[elem][j] + self.A[j][elem] for j in range(self.size))

    def greedySolution(self) -> Solution:
        sol = Solution()
        sol.cost = 0.0
        covered = BitSet(self.size)

        candidates = list(range(self.size))
        costs = [self.getValue(i) for i in candidates]
        candidates.sort(key=lambda x: costs[x])

        elem_to_add = 0
        added = 0
        while costs[candidates[0]] > 0:
            if covered.cardinality() >= self.size // 2:
                for i in range(self.size):
                    if costs[i] == float("-inf"):
                        continue
                    costs[i] = self.evaluateInsertionCost(i, sol)
                candidates.sort(key=lambda x: costs[x])
                elem_to_add = candidates[0]
            else:
                elem_to_add = candidates[added]
                added += 1

            costs[elem_to_add] = float("-inf")
            delta_cost = self.evaluateInsertionCost(elem_to_add, sol)
            sol.append(elem_to_add)
            covered.or_(self.coverBits[elem_to_add])
            sol.cost += delta_cost
        return sol

    def fixSolutionGreedy(self, sol: Solution) -> Solution:
        covered = self.coveredOf(sol)
        uncovered = covered.clone()
        uncovered.flip(0, self.size)

        while uncovered.cardinality() > 0:
            bestSet, bestCover = -1, 0
            for i in range(self.size):
                if i in sol:
                    continue
                bs = self.coverBits[i].clone()
                bs.and_(uncovered)
                c = bs.cardinality()
                if c > bestCover:
                    bestCover, bestSet = c, i

            if bestSet == -1:
                print("No set can cover uncovered elements")
                return None

            delta_cost = self.evaluateInsertionCost(bestSet, sol)
            sol.append(bestSet)
            covered.or_(self.coverBits[bestSet])
            uncovered = covered.clone()
            uncovered.flip(0, self.size)
            sol.cost += delta_cost

        if covered.cardinality() < self.size:
            print("Not all elements are covered")
        return sol

    def varToSolution(self, vars: List[float]) -> Solution:
        return Solution([i for i, v in enumerate(vars) if v > 0.5])

    def evaluateMAXSCQBF(self) -> float:
        total = 0.0
        for i in range(self.size):
            aux = sum(self.variables[j] * self.A[i][j] for j in range(self.size))
            total += aux * self.variables[i]
        notCovered = self.size - self.coveredOf(self.varToSolution(self.variables)).cardinality()
        return total - self.lambda_ * notCovered

    def evaluateInsertionCost(self, elem: int, sol: Solution) -> float:
        self.setVariables(sol)
        dQ = self.evaluateInsertionMAXSCQBF(elem)
        newlyCovered = self.newlyCoveredBy(elem, sol)
        return 1 if (dQ <= 0 and newlyCovered > 0) else dQ

    def evaluateInsertionMAXSCQBF(self, i: int) -> float:
        if self.variables[i] == 1:
            return 0.0
        return self.evaluateContributionMAXSCQBF(i)

    def evaluateRemovalCost(self, elem: int, sol: Solution) -> float:
        self.setVariables(sol)
        dQ = self.evaluateRemovalMAXSCQBF(elem)
        newlyUncovered = self.newlyUncoveredBy(elem, sol)
        return float("-inf") if newlyUncovered > 0 else dQ

    def evaluateRemovalMAXSCQBF(self, i: int) -> float:
        if self.variables[i] == 0:
            return 0.0
        return -self.evaluateContributionMAXSCQBF(i)

    def evaluateExchangeCost(self, elemIn: int, elemOut: int, sol: Solution) -> float:
        self.setVariables(sol)
        dQ = self.evaluateExchangeMAXSCQBF(elemIn, elemOut)
        newlyCoveredIn = self.newlyCoveredBy(elemIn, sol)
        newlyUncoveredOut = self.newlyUncoveredByConsideringExchange(elemIn, elemOut, sol)
        if newlyUncoveredOut > 0:
            return float("-inf")
        return 1 if (dQ <= 0 and newlyCoveredIn > 0) else dQ

    def evaluateExchangeMAXSCQBF(self, in_idx: int, out_idx: int) -> float:
        if in_idx == out_idx:
            return 0.0
        if self.variables[in_idx] == 1:
            return self.evaluateRemovalMAXSCQBF(out_idx)
        if self.variables[out_idx] == 0:
            return self.evaluateInsertionMAXSCQBF(in_idx)

        total = self.evaluateContributionMAXSCQBF(in_idx) - self.evaluateContributionMAXSCQBF(out_idx)
        total -= self.A[in_idx][out_idx] + self.A[out_idx][in_idx]
        return total

    def evaluateContributionMAXSCQBF(self, i: int) -> float:
        total = sum(self.variables[j] * (self.A[i][j] + self.A[j][i]) for j in range(self.size) if i != j)
        total += self.A[i][i]
        return total

    def readInput(self, filename: str) -> int:
        """Reads the instance file and fills matrices A and S."""
        with open(filename, "r") as f:
            tokens = re.findall(r"[-+]?[0-9]*\.?[0-9]+", f.read())

        idx = 0
        n = int(float(tokens[idx])); idx += 1
        self.A = [[0.0]*n for _ in range(n)]
        self.S = [None]*n

        # Read set sizes
        for i in range(n):
            self.S[i] = [0]*int(float(tokens[idx])); idx += 1

        # Read sets
        for i in range(n):
            for j in range(len(self.S[i])):
                self.S[i][j] = int(float(tokens[idx])); idx += 1

        outeri = 0
        outerj = 0

        # Read A matrix
        try:
            for i in range(n):
                for j in range(i,n):
                    outeri = i
                    outerj = j

                    self.A[i][j] = float(tokens[idx]); idx += 1
        except IndexError:
            print(f"Error reading A matrix from file in line:{outeri}, {outerj}, {filename}")
            raise

        # Determine variable indexing offset (0- or 1-based)
        hasZero = any(0 in s for s in self.S)
        shift = 0 if hasZero else -1

        # Build coverBits
        self.coverBits = []
        for i in range(n):
            bs = BitSet(n)
            for v in self.S[i]:
                idxv = v + shift
                if 0 <= idxv < n:
                    bs.set(idxv)
            self.coverBits.append(bs)

        self.lambda_ = 10000.0
        return n

    def coveredOf(self, sol: Solution) -> BitSet:
        covered = BitSet(self.size)
        for idx in sol:
            covered.or_(self.coverBits[idx])
        return covered

    def coverCountOf(self, sol: Solution):
        counts = [0]*self.size
        for i in sol:
            bs = self.coverBits[i]
            k = bs.nextSetBit(0)
            while k >= 0:
                counts[k] += 1
                k = bs.nextSetBit(k+1)
        return counts

    def newlyCoveredBy(self, elem: int, sol: Solution):
        uncovered = self.coveredOf(sol)
        uncovered.flip(0, self.size)
        bs = self.coverBits[elem].clone()
        bs.and_(uncovered)
        return bs.cardinality()

    def newlyUncoveredBy(self, elem: int, sol: Solution):
        counts = self.coverCountOf(sol)
        out = self.coverBits[elem]
        return sum(1 for k in out._set if counts[k] == 1)

    def newlyUncoveredByConsideringExchange(self, in_idx: int, out_idx: int, sol: Solution):
        counts = self.coverCountOf(sol)
        out, inn = self.coverBits[out_idx], self.coverBits[in_idx]
        return sum(1 for k in out._set if counts[k] == 1 and not inn.get(k))

    def allocateVariables(self):
        return [0.0]*self.size

    def resetVariables(self):
        for i in range(self.size):
            self.variables[i] = 0.0

    def printMatrix(self):
        for i in range(self.size):
            for j in range(i, self.size):
                print(self.A[i][j], end=" ")
            print()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python MAX_SC_QBF.py <instance_file>")
    else:
        problem = MAX_SC_QBF(sys.argv[1])
        sol = Solution([19, 24, 4, 11, 8, 0, 23, 3, 5, 20, 7, 1, 14, 9, 22, 15, 17])
        problem.evaluate(sol)
        print("Solution:", sol)
        print("Value:", sol.cost)
        print("Covered:", problem.coveredOf(sol))
