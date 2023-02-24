# -*- coding: utf-8 -*-
from configs import get_config
from solver import Solver
from data_loader import get_loader


if __name__ == '__main__':
    """ Main function that sets the data loaders; trains and evaluates the model. """
    config = get_config(mode='train')
    test_config = get_config(mode='test')

    print(config)
    print(test_config)

    train_loader = get_loader(config.mode, config.video_type, config.split_index)
    test_loader = get_loader(test_config.mode, test_config.video_type, test_config.split_index)
    solver = Solver(config, train_loader, test_loader)

    solver.build()       # build the model
    solver.evaluate(-1)  # evaluate the randomly initialized (untrained) network
    solver.train()       # train the model
