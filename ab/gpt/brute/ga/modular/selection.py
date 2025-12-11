from abc import ABC, abstractmethod
import random

class SelectionStrategy(ABC):
    @abstractmethod
    def select(self, population):
        pass

class TournamentSelection(SelectionStrategy):
    def __init__(self, tournament_size=3):
        self.tournament_size = tournament_size

    def select(self, population):
        # Ensure we don't try to sample more than we have
        actual_size = min(self.tournament_size, len(population))
        competitors = random.sample(population.individuals, actual_size)
        # Return the one with the highest fitness
        # Handle None fitness safely
        winner = max(competitors, key=lambda x: x.fitness if x.fitness is not None else -float('inf'))
        return winner
