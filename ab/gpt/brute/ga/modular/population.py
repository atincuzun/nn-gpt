import random
from .individual import Individual

class Population:
    def __init__(self, size):
        self.size = size
        self.individuals = []

    def initialize(self, search_space):
        self.individuals = []
        for _ in range(self.size):
            chromosome = {}
            for key, values in search_space.items():
                chromosome[key] = random.choice(values)
            self.individuals.append(Individual(chromosome))
        print(f"Initialized population with {self.size} individuals.")

    def add(self, individual):
        self.individuals.append(individual)

    def sort_by_fitness(self, descending=True):
        # Handle None fitness safely by treating it as -1 (or very low value)
        self.individuals.sort(
            key=lambda x: x.fitness if x.fitness is not None else -float('inf'),
            reverse=descending
        )

    def get_best(self):
        self.sort_by_fitness()
        if not self.individuals:
            return None
        return self.individuals[0]

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, index):
        return self.individuals[index]
