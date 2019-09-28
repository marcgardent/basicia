
from basiciaevolver import BasiciaEvolver
from DeepEvolve.utils import AllPossibleGenesBuilder, GenomeReader
import logging
import random;

def dummy_train_and_score_callback(geneparam):
    """
    geneparam, for instance:
    """
    logging.debug("***Train and score***");
    g = GenomeReader(geneparam)
    logging.debug(g);
    return random.random()

def main():
    
    b= AllPossibleGenesBuilder()
    b.set('actor.layer_count', [1, 2, 3, 4, 5])
    b.duplicate('actor.layer._neurons', [8, 16, 32, 64, 128, 256, 512, 768, 1024], 6)

    b.set('actor.activation', ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'])
    b.set('actor.optimizer',  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam'])

    b.duplicate('critic.nb_neurons', [8, 16, 32, 64, 128, 256, 512, 768, 1024], 6)
    b.set('critic.activation_actor', ['relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'])
    b.set('critic.optimizer_actor',  ['rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam'])

    logging.debug(b.toDict())
    blackbox = BasiciaEvolver(dummy_train_and_score_callback, b.toDict(), 30);
    blackbox.tournament(30);

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.DEBUG)

    main()