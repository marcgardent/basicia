from DeepEvolve import Evolver
import logging
import random;
# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO
)

class DeepEvolveParams:
    
    def __init__(self):
        self.population = 30
        self.all_possible_genes = {
            'nb_neurons': [8, 16, 32, 64, 128, 256, 512, 768, 1024],
            'nb_layers':  [1, 2, 3, 4 ,5],
            'activation': ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'],
            'optimizer':  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam']
        }

        # replace nb_neurons with 1 unique value for each layer
        # 6th value reserved for dense layer <--- TODO avoid hardcode!
        nb_neurons = self.all_possible_genes['nb_neurons']
        for i in range(1, 7):
          self.all_possible_genes['nb_neurons_' + str(i)] = nb_neurons
        # remove old value from dict
        self.all_possible_genes.pop('nb_neurons')

def dummy_train_and_score_callback(geneparam):
    """
    geneparam, for instance:
        - {'nb_layers': 5, 'activation': 'hard_sigmoid', 'optimizer': 'adamax', 'nb_neurons': [16, 256, 16, 128, 256, 8]}
    """
    return random.random()

class BasiciaEvolver:
    def __init__(self, deepEvolveParams):
        self.generationIndex =0;
        self.deepEvolveParams = deepEvolveParams;
        self.evolver = Evolver(self.deepEvolveParams.all_possible_genes)
        self.genomes = self.evolver.create_population(self.deepEvolveParams.population)

    def evolve(self):
        logging.info(f"***Now in generation {self.generationIndex}***")
        self.print_genomes(self.genomes);

        for genome in self.genomes:
            genome.train(dummy_train_and_score_callback)

        self.genomes = self.evolver.evolve(self.genomes)

    def evolve_over_time(self,generations):
        for i in range(0, generations):
            self.evolve()
        
        # Sort our final population according to performance.
        self.genomes = sorted(self.genomes, key=lambda x: x.accuracy, reverse=True)
        logging.info(f"***Top 5***")
        # Print out the top 5 networks/genomes.
        self.print_genomes(self.genomes[:5])

    def print_genomes(self, genomes):
        logging.info('-'*80)
        for genome in genomes:
            genome.print_genome()

def main():
    params = DeepEvolveParams();
    blackbox = BasiciaEvolver(params);
    blackbox.evolve_over_time(30);


if __name__ == "__main__":
    # execute only if run as a script
    main()