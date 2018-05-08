# -*- coding: utf-8 -*-
"""Unit test for the MRT objective"""

from nmt import train

import os
import unittest
import shutil
import logging
import numpy

NEMATUS = os.path.abspath(os.path.join(__file__, "../.."))
DATA_DIR = NEMATUS + "/data_unittest/"
LOG_PREFIX = DATA_DIR + "toy"
WORD_REWARD = DATA_DIR + "toy.token_level"
BASE_MODEL = DATA_DIR + "base_model/"
DATA_SETS = [DATA_DIR + '/toy.pre', LOG_PREFIX + '.tgt']
DICTIONARIES = [DATA_DIR + '/toy.pre.json', DATA_DIR + '/toy.lin.json']


class MRTObjectiveTests(unittest.TestCase):

    def setUp(self):
        self.tolerance = 1e-6

    def test_MRT(self):
        """
        Verifies that the correct cost is calculated for MRT
        :return: 0 on success
        """
        logging.info("Starting Test for MRT..")
        working_dir = DATA_DIR + "MRT/"
        if os.path.isdir(working_dir):
            shutil.rmtree(working_dir)
        shutil.copytree(BASE_MODEL, working_dir)
        cost = train(saveto="%smodel.npz" % working_dir,
                     reload_=True,
                     shuffle_each_epoch=False,
                     datasets=DATA_SETS,
                     dictionaries=DICTIONARIES,
                     objective='MRT',
                     mrt_samples_meanloss=0,
                     unittest=True)

        true_cost = -1.0
        numpy.testing.assert_almost_equal(true_cost, cost)
        shutil.rmtree(working_dir)
        logging.info("Finished Test for MRT..")
        return 0


def main():
    """
    Runs all unit tests
    :return: 0 on success
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info("Starting Unit Tests for MRT objectives..")
    unittest.main()
    return 0


if __name__ == "__main__":
    main()
