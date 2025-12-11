from abc import ABC, abstractmethod
import random

class CrossoverStrategy(ABC):
    @abstractmethod
    def crossover(self, parent1, parent2):
        pass

class SinglePointCrossover(CrossoverStrategy):
    def crossover(self, parent1, parent2):
        child_chromo = {}
        genes = list(parent1.chromosome.keys())
        
        if len(genes) > 1:
            crossover_point = random.randint(1, len(genes) - 1)
        else:
            crossover_point = 1 # Fallback for single gene

        for i, gene in enumerate(genes):
            if i < crossover_point:
                child_chromo[gene] = parent1.chromosome[gene]
            else:
                child_chromo[gene] = parent2.chromosome[gene]
        
        return child_chromo
