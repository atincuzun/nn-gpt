from abc import ABC, abstractmethod
import random

class MutationStrategy(ABC):
    def __init__(self, mutation_rate):
        self.mutation_rate = mutation_rate

    @abstractmethod
    def mutate(self, chromosome, search_space):
        pass

class RandomResettingMutation(MutationStrategy):
    def mutate(self, chromosome, search_space):
        mutated_chromo = chromosome.copy()
        for gene in search_space.keys():
            if random.random() < self.mutation_rate:
                current_value = mutated_chromo[gene]
                possible_values = search_space[gene]
                # Exclude current value if possible
                other_values = [v for v in possible_values if v != current_value]
                if other_values:
                    mutated_chromo[gene] = random.choice(other_values)
        return mutated_chromo
