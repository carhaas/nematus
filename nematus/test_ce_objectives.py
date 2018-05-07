# -*- coding: utf-8 -*-
"""Unit test for various objectives"""

import os
import unittest
import shutil
import logging
import numpy as np

VOCAB_SIZE = 90000
SRC = "pre"
TGT = "lin"
NEMATUS = os.path.abspath(os.path.join(__file__, "../.."))
DATA_DIR = NEMATUS + "/data_unittest/"
LOG_PREFIX = DATA_DIR + "/toy"
BASE_MODEL = DATA_DIR + "base_model/"

from nmt import train


class CEObjectiveTests(unittest.TestCase):

    def setUp(self):
        self.tolerance = 1e-6

    def test_CE(self):
        """
        Verifies that the correct cost is calculated for IPS with reweigh.
        :return: 0 on success
        """
        logging.info("Starting Test: test_CE..")
        working_dir = DATA_DIR + "CE/"
        if os.path.isdir(working_dir):
            shutil.rmtree(working_dir)
        shutil.copytree(BASE_MODEL, working_dir)
        cost = train(saveto="%smodel.npz" % working_dir,
            reload_=True,
            patience=50,
            dim_word=1000,
            dim=1024,
            shuffle_each_epoch=False,
            lrate=0.0001,
            optimizer='adadelta',
            maxlen=201,
            batch_size=10,
            valid_batch_size=10,
            datasets=[DATA_DIR + '/toy.' + SRC, LOG_PREFIX + '.tgt'],
            dictionaries=[DATA_DIR + '/toy.' + SRC + '.json', DATA_DIR + '/toy.' + TGT + '.json'],
            objective='CE',
            unittest=True)

        true_cost = 0.2190007519173402
        np.testing.assert_almost_equal(true_cost, cost)
        shutil.rmtree(working_dir)
        return 0

def main():
    """
    Runs all unit tests
    :return: 0 on success
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info("Starting Unit Tests for CE objective..")
    unittest.main()
    return 0

if __name__ == "__main__":
    main()