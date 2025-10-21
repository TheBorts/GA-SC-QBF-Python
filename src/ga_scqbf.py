import random
import time
from typing import List
from AbstractGA import AbstractGA
from Max_SC_QBF import MAX_SC_QBF
from Solution import Solution


random_seed = 0
rng = random.Random(random_seed)


class GA_SCQBF(AbstractGA):
    """
    Python implementation of GA_SCQBF (Genetic Algorithm for the Set Cover QBF problem).
    Translated faithfully from the Java version.
    """

    def __init__(self, popSize: int, mutationRate: float, filename: str,
                 enableLatinHyperCube: bool, enableMutateOrCrossover: bool,
                 enableUniformCrossover: bool, timeLimit: int, bestKnown: float):
        super().__init__(MAX_SC_QBF(filename), 1, popSize, mutationRate)
        self.enableLatinHyperCube = enableLatinHyperCube
        self.enableMutateOrCrossover = enableMutateOrCrossover
        self.enableUniformCrossover = enableUniformCrossover
        self.startTime = int(time.time() * 1000)
        self.bestKnown = bestKnown
        self.timeLimit = timeLimit

    def createEmptySol(self):
        sol = Solution()
        sol.cost = 0.0
        return sol

    def createGreedySol(self):
        return self.ObjFunction.greedySolution()

    def fixChromosome(self, chromosome):
        sol = self.decode(chromosome)
        sol = self.ObjFunction.fixSolutionGreedy(sol)
        chromosome = self.code(sol)
        return chromosome

    def code(self, sol: Solution):
        chromosome = AbstractGA.Chromosome()
        for _ in range(self.chromosomeSize):
            chromosome.append(0)
        for var in sol:
            chromosome[var] = 1
        return chromosome

    def decode(self, chromosome):
        solution = self.createEmptySol()
        for locus, value in enumerate(chromosome):
            if value == 1:
                solution.append(locus)
        self.ObjFunction.evaluate(solution)
        return solution

    def generateRandomChromosome(self):
        chromosome = AbstractGA.Chromosome()
        for _ in range(self.chromosomeSize):
            chromosome.append(rng.randint(0, 1))
        chromosome = self.fixChromosome(chromosome)
        return chromosome

    def fixPopulation(self, population):
        fixedPop = AbstractGA.Population()
        for chromosome in population:
            fixedPop.append(self.fixChromosome(chromosome))
        return fixedPop

    def uniform_crossover(self, parents):
        offsprings = AbstractGA.Population()
        for i in range(0, self.popSize, 2):
            off_mask = []
            parent1 = parents[i]
            parent2 = parents[i + 1]
            for j in range(self.chromosomeSize):
                off_mask.append(0 if j < self.chromosomeSize // 2 else 1)
            random.shuffle(off_mask)

            offspring1 = AbstractGA.Chromosome()
            offspring2 = AbstractGA.Chromosome()
            for j in range(self.chromosomeSize):
                if off_mask[j] == 0:
                    offspring1.append(parent1[j])
                    offspring2.append(parent2[j])
                else:
                    offspring1.append(parent2[j])
                    offspring2.append(parent1[j])

            offsprings.append(offspring1)
            offsprings.append(offspring2)
        return offsprings

    def crossover(self, parents):
        if self.enableUniformCrossover:
            return self.uniform_crossover(parents)

        offsprings = AbstractGA.Population()
        for i in range(0, self.popSize, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1]
            crosspoint1 = rng.randint(0, self.chromosomeSize)
            crosspoint2 = rng.randint(0, self.chromosomeSize - crosspoint1) + crosspoint1

            offspring1 = AbstractGA.Chromosome()
            offspring2 = AbstractGA.Chromosome()
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

    def fitness(self, chromosome):
        return self.decode(chromosome).cost

    def mutateGene(self, chromosome, locus: int):
        chromosome[locus] = 1 - (chromosome[locus] % 2)
        return chromosome

    def generateLatinHypercubePopulation(self):
        population = AbstractGA.Population()
        for _ in range(self.popSize):
            population.append(AbstractGA.Chromosome())

        for locus in range(self.chromosomeSize):
            values = [i % 2 for i in range(self.popSize)]
            random.shuffle(values)
            for i in range(self.popSize):
                population[i].append(values[i])
        return population

    def initializePopulation(self):
        if self.enableLatinHyperCube:
            population = self.generateLatinHypercubePopulation()
        else:
            population = AbstractGA.Population()
            while len(population) < self.popSize:
                population.append(self.generateRandomChromosome())

        greedyChromosome = self.code(self.createGreedySol())
        worstChromosome = self.getWorstChromosome(population)
        if worstChromosome in population:
            population.remove(worstChromosome)
        population.append(greedyChromosome)
        return population

    def solve(self):
        population = self.initializePopulation()
        self.bestChromosome = self.getBestChromosome(population)
        self.bestSol = self.decode(self.bestChromosome)
        cover = self.ObjFunction.coveredOf(self.bestSol).cardinality()

        print(f"Best known = {self.bestKnown}")
        print(f"(Gen. 0) BestSol = {self.bestSol}")
        print(f"Real value = {self.ObjFunction.evaluate(self.bestSol)} Cover = {cover}")

        generation = 0
        while (int(time.time() * 1000) - self.startTime) < self.timeLimit and self.bestSol.cost < self.bestKnown:
            parents = self.selectParents(population)
            offsprings = parents
            mutants = parents

            if self.enableMutateOrCrossover:
                crossoverProbability = 0.9
                mutationProbability = 0.3
                self.mutationRate = 0.05

                if rng.random() < crossoverProbability:
                    offsprings = self.crossover(parents)
                if rng.random() < mutationProbability:
                    mutants = self.mutate(offsprings)
                else:
                    mutants = offsprings

                if (int(time.time() * 1000) - self.startTime) > self.timeLimit * 0.7:
                    crossoverProbability = 0.2
                    mutationProbability = 1.0
                    self.mutationRate = 0.1
            else:
                offsprings = self.crossover(parents)
                mutants = self.mutate(offsprings)

            population = self.selectPopulation(mutants)
            self.bestChromosome = self.getBestChromosome(population)

            if self.fitness(self.bestChromosome) > self.bestSol.cost:
                self.bestSol = self.decode(self.bestChromosome)
                if self.verbose:
                    elapsed = (int(time.time() * 1000) - self.startTime) / 1000.0
                    print(f"(Gen. {generation} at {elapsed}s) BestSol = {self.bestSol}")
                    cover = self.ObjFunction.coveredOf(self.bestSol).cardinality()
                    print(f"Real value = {self.ObjFunction.evaluate(self.bestSol)} Cover = {cover}")

            generation += 1
        return self.bestSol


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="GA for Set Cover QBF")
    parser.add_argument("--instance", required=True)
    parser.add_argument("--enable-latin-hyper-cube", required=True)
    parser.add_argument("--enable-mutate-or-crossover", required=True)
    parser.add_argument("--enable-uniform-crossover", required=True)
    parser.add_argument("--tl", required=True, type=int)
    parser.add_argument("--population-size", required=True, type=int)
    parser.add_argument("--mutation-rate", required=True, type=float)
    args = parser.parse_args()

    known_results = {
        "instances/instancia_n25_padrao1.txt": 1441.0,
        "instances/instancia_n25_padrao2.txt": 2030.0,
        "instances/instancia_n25_padrao3.txt": 975.0,
        "instances/instancia_n50_padrao1.txt": 6144.0,
        "instances/instancia_n50_padrao2.txt": 6419.0,
        "instances/instancia_n50_padrao3.txt": 7746.0,
    }

    filename = args.instance
    enableLatinHyperCube = args.enable_latin_hyper_cube.lower() == "true"
    enableMutateOrCrossover = args.enable_mutate_or_crossover.lower() == "true"
    enableUniformCrossover = args.enable_uniform_crossover.lower() == "true"
    timeLimit = args.tl * 1000
    populationSize = args.population_size
    mutationRate = args.mutation_rate

    startTime = int(time.time() * 1000)
    ga = GA_SCQBF(
        populationSize, mutationRate, filename,
        enableLatinHyperCube, enableMutateOrCrossover,
        enableUniformCrossover, timeLimit,
        known_results.get(filename, float("inf"))
    )
    bestSol = ga.solve()
    print("maxVal =", bestSol)
    endTime = int(time.time() * 1000)
    totalTime = endTime - startTime
    print("TotalTime =", totalTime / 1000.0, "seconds")
