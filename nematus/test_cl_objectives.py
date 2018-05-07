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


class CLObjectiveTests(unittest.TestCase):

    def setUp(self):
        self.tolerance = 1e-6

    def test_IPS_reweigh_word_level(self):
        """
        Verifies that the correct cost is calculated for IPS with reweigh.
        :return: 0 on success
        """
        logging.info("Starting Test: test_IPS_reweigh_word_level..")
        working_dir = DATA_DIR + "IPS_w_r/"
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
            objective='CL',
            cl_deterministic=True,
            cl_log=LOG_PREFIX + '.json',
            cl_reweigh=True,
            cl_external_reward=DATA_DIR + '/toy.word_level',
            cl_word_rewards=True,
            unittest=True)

        true_cost = -1.1665413125898798
        np.testing.assert_almost_equal(true_cost, cost)
        shutil.rmtree(working_dir)
        return 0

    def test_IPS_word_level(self):
        """
        Verifies that the correct cost is calculated for IPS with reweigh.
        :return: 0 on success
        """
        logging.info("Starting Test: test_IPS_word_level..")
        working_dir = DATA_DIR + "IPS_w/"
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
            objective='CL',
            cl_deterministic=True,
            cl_log=LOG_PREFIX + '.json',
            cl_external_reward=DATA_DIR + '/toy.word_level',
            cl_word_rewards=True,
            unittest=True)

        true_cost = -0.9641428801812324
        np.testing.assert_almost_equal(true_cost, cost)
        shutil.rmtree(working_dir)
        return 0

    def test_IPS_reweigh_seq_level(self):
        """
        Verifies that the correct cost is calculated for IPS with reweigh.
        :return: 0 on success
        """
        logging.info("Starting Test: test_IPS_reweigh_seq_level..")
        working_dir = DATA_DIR + "IPS_s_r/"
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
            objective='CL',
            cl_deterministic=True,
            cl_log=LOG_PREFIX + '.json',
            cl_reweigh=True,
            unittest=True)

        true_cost = -0.8526537389124744
        np.testing.assert_almost_equal(true_cost, cost)
        shutil.rmtree(working_dir)
        return 0

    def test_IPS_seq_level(self):
        """
        Verifies that the correct cost is calculated for IPS with reweigh.
        :return: 0 on success
        """
        logging.info("Starting Test: test_IPS_seq_level..")
        working_dir = DATA_DIR + "IPS_s/"
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
            objective='CL',
            cl_deterministic=True,
            cl_log=LOG_PREFIX + '.json',
            unittest=True)

        true_cost = -0.7047157462492611
        np.testing.assert_almost_equal(true_cost, cost)
        shutil.rmtree(working_dir)
        return 0

    def test_IPS_seq_level_propensity(self):
        """
        Verifies that the correct cost is calculated for IPS with reweigh.
        :return: 0 on success
        """
        logging.info("Starting Test: test_IPS_seq_level..")
        working_dir = DATA_DIR + "IPS_s/"
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
            objective='CL',
            cl_deterministic=False,
            cl_log=LOG_PREFIX + '.json',
            unittest=True)

        true_cost = -0.8489172076113973
        np.testing.assert_almost_equal(true_cost, cost)
        shutil.rmtree(working_dir)
        return 0

def main():
    """
    Runs all unit tests
    :return: 0 on success
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info("Starting Unit Tests for CL objectives..")
    unittest.main()
    return 0

if __name__ == "__main__":
    main()