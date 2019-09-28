import logging
import random
from basicia.DeepEvolve.models import *
from basicia.DeepEvolve import Producter

def dummy_train_and_score_callback(ddpgagent):
    """
    geneparam, for instance:
    """
    logging.debug("***Train and score***");
    logging.debug(ddpgagent.toString());
    logging.debug(ddpgagent['actor']['optimizer'].value)
    logging.debug(ddpgagent['actor']['layers'][0]['neurons'].value)

    return random.random()

def main():
    
    # For Instance : 
    neurons = DiscretGene((8, 16, 32, 64, 128, 256, 512, 768, 1024))
    activation = DiscretGene(('relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid','softplus','linear'))
    optimizer = DiscretGene(('rmsprop', 'adam', 'sgd', 'adagrad','adadelta', 'adamax', 'nadam'))
    layer = ComplexNode({
        "activation" : activation.clone(),
        "neurons" : neurons.clone()
        })

    nn = ComplexNode({
            "layers" : ArrayNode.Duplicate(layer, 6),
            "optimizer" : optimizer.clone()
        })
    
    ddpgagent = ComplexNode({
        "actor" : nn.clone(),
        "critic" : nn.clone()
    })

    ddpgagent.set_genes_random()
    tuple(ddpgagent.all_genes())
    tuple(ddpgagent.all_mutables())
    print(ddpgagent.toString())

    p =Producter(dummy_train_and_score_callback, ddpgagent, 30)
    p.tournament(50)

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.DEBUG
    )
    main()