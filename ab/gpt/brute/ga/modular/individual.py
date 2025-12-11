class Individual:
    def __init__(self, chromosome, fitness=None):
        self.chromosome = chromosome
        self.fitness = fitness

    def __repr__(self):
        return f"Individual(fitness={self.fitness}, chromosome={self.chromosome})"
