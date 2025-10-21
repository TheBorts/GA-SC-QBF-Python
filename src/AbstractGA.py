import random
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic
from Solution import Solution
from Evaluator import Evaluator

G = TypeVar("G", bound=float)
F = TypeVar("F")


class AbstractGA(ABC, Generic[G, F]):
    """Abstract class for a Genetic Algorithm (GA)."""

    verbose = True
    rng = random.Random(0)

    class Chromosome(list):
        """Represents a chromosome as a list of genes."""
        pass

    class Population(list):
        """Represents a population of chromosomes."""
        pass

    def __init__(self, objFunction: Evaluator, generations: int, popSize: int, mutationRate: float):
        self.ObjFunction = objFunction
        self.generations = generations
        self.popSize = popSize
        self.chromosomeSize = self.ObjFunction.getDomainSize()
        self.mutationRate = mutationRate

        self.bestCost = None
        self.bestSol: Solution[F] = None
        self.bestChromosome: AbstractGA.Chromosome = None

    # --- Abstract methods to be implemented by subclasses ---

    @abstractmethod
    def createEmptySol(self) -> Solution[F]:
        pass

    @abstractmethod
    def decode(self, chromosome: "AbstractGA.Chromosome") -> Solution[F]:
        pass

    @abstractmethod
    def generateRandomChromosome(self) -> "AbstractGA.Chromosome":
        pass

    @abstractmethod
    def fitness(self, chromosome: "AbstractGA.Chromosome") -> float:
        pass

    @abstractmethod
    def mutateGene(self, chromosome: "AbstractGA.Chromosome", locus: int) -> "AbstractGA.Chromosome":
        pass

    # --- Core GA methods ---

    def solve(self) -> Solution[F]:
        """Runs the main Genetic Algorithm loop."""
        population = self.initializePopulation()

        self.bestChromosome = self.getBestChromosome(population)
        self.bestSol = self.decode(self.bestChromosome)
        print(f"(Gen. 0) BestSol = {self.bestSol}")

        for g in range(1, self.generations + 1):
            parents = self.selectParents(population)
            offsprings = self.crossover(parents)
            mutants = self.mutate(offsprings)
            newPopulation = self.selectPopulation(mutants)
            population = newPopulation

            self.bestChromosome = self.getBestChromosome(population)
            if self.fitness(self.bestChromosome) > self.bestSol.cost:
                self.bestSol = self.decode(self.bestChromosome)
                if self.verbose:
                    print(f"(Gen. {g}) BestSol = {self.bestSol}")

        return self.bestSol

    def initializePopulation(self) -> "AbstractGA.Population":
        """Randomly generates the initial population."""
        population = self.Population()
        while len(population) < self.popSize:
            chromosome = self.generateRandomChromosome()
            population.append(chromosome)
        return population

    def getBestChromosome(self, population: "AbstractGA.Population") -> "AbstractGA.Chromosome":
        """Returns the chromosome with the highest fitness."""
        bestFitness = float("-inf")
        bestChromosome = None
        for c in population:
            fit = self.fitness(c)
            if fit > bestFitness:
                bestFitness = fit
                bestChromosome = c
        return bestChromosome

    def getWorstChromosome(self, population: "AbstractGA.Population") -> "AbstractGA.Chromosome":
        """Returns the chromosome with the lowest fitness."""
        worstFitness = float("inf")
        worstChromosome = None
        for c in population:
            fit = self.fitness(c)
            if fit < worstFitness:
                worstFitness = fit
                worstChromosome = c
        return worstChromosome

    def selectParents(self, population: "AbstractGA.Population") -> "AbstractGA.Population":
        """Selects parents using the tournament method."""
        parents = self.Population()
        while len(parents) < self.popSize:
            index1 = self.rng.randint(0, self.popSize - 1)
            index2 = self.rng.randint(0, self.popSize - 1)
            parent1 = population[index1]
            parent2 = population[index2]
            if self.fitness(parent1) > self.fitness(parent2):
                parents.append(parent1)
            else:
                parents.append(parent2)
        return parents

    def crossover(self, parents: "AbstractGA.Population") -> "AbstractGA.Population":
        """Performs 2-point crossover."""
        offsprings = self.Population()
        for i in range(0, self.popSize, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]

            crosspoint1 = self.rng.randint(0, self.chromosomeSize)
            crosspoint2 = self.rng.randint(crosspoint1, self.chromosomeSize)

            offspring1 = self.Chromosome()
            offspring2 = self.Chromosome()

            for j in range(self.chromosomeSize):
                if crosspoint1 <= j < crosspoint2:
                    offspring1.append(parent2[j])
                    offspring2.append(parent1[j])
                else:
                    offspring1.append(parent1[j])
                    offspring2.append(parent2[j])

            offsprings.append(offspring1)
            offsprings.append(offspring2)

        return offsprings

    def mutate(self, offsprings: "AbstractGA.Population") -> "AbstractGA.Population":
        """Mutates each gene with probability 'mutationRate'."""
        for c in offsprings:
            for locus in range(self.chromosomeSize):
                if self.rng.random() < self.mutationRate:
                    self.mutateGene(c, locus)
        return offsprings

    def selectPopulation(self, offsprings: "AbstractGA.Population") -> "AbstractGA.Population":
        """Elitist replacement: replaces the worst with the best from the previous gen."""
        worst = self.getWorstChromosome(offsprings)
        if self.fitness(worst) < self.fitness(self.bestChromosome):
            offsprings.remove(worst)
            offsprings.append(self.bestChromosome)
        return offsprings
