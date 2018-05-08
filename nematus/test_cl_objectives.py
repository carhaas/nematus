#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit test for various counterfactual objectives"""

from nmt import train

import os
import unittest
import shutil
import logging
import numpy

NEMATUS = os.path.abspath(os.path.join(__file__, "../.."))
DATA_DIR = NEMATUS + "/data_unittest/"
LOG_PREFIX = DATA_DIR + "toy"
WORD_REWARD = DATA_DIR + "toy.word_level"
BASE_MODEL = DATA_DIR + "base_model/"
DATA_SETS = [DATA_DIR + '/toy.pre', LOG_PREFIX + '.tgt']
DICTIONARIES = [DATA_DIR + '/toy.pre.json', DATA_DIR + '/toy.lin.json']

def prepare_base_model(working_dir):
    if os.path.isdir(working_dir):
        shutil.rmtree(working_dir)
    shutil.copytree(BASE_MODEL, working_dir)


class CLObjectiveTests(unittest.TestCase):

    def setUp(self):
        self.tolerance = 1e-6

    def test_DPM(self):
        """
        Verifies that the correct cost is calculated for DPM.
        :return: 0 on success
        """
        logging.info("Starting Test for DPM..")
        working_dir = DATA_DIR + "DPM/"
        prepare_base_model(working_dir)
        cost, prepared_rewards, prepared_word_propensities, reweigh_sum = \
            train(saveto="%smodel.npz" % working_dir,
                  reload_=True,
                  shuffle_each_epoch=False,
                  datasets=DATA_SETS,
                  dictionaries=DICTIONARIES,
                  objective='CL',
                  cl_deterministic=True,
                  cl_log=LOG_PREFIX + ".json",
                  unittest=True)

        true_cost = -0.7047157462492611
        self.assertAlmostEqual(true_cost, cost)
        true_prepared_rewards = numpy.array([1., 0.8222672, 1., 0.8274377, 0.7813821, 0.7813821, 0.6673543, 1., 0.6093617, 1.])
        numpy.testing.assert_almost_equal(true_prepared_rewards, prepared_rewards)
        true_prepared_word_propensities = numpy.zeros(shape=(22, 10))
        numpy.testing.assert_almost_equal(true_prepared_word_propensities, prepared_word_propensities)
        true_reweigh_sum = 0.0
        numpy.testing.assert_almost_equal(true_reweigh_sum, reweigh_sum)
        shutil.rmtree(working_dir)
        logging.info("Finished Test for DPM..")
        return 0

    def test_DPM_OSL(self):
        """
        Verifies that the correct cost is calculated for DPM+OSL.
        :return: 0 on success
        """
        logging.info("Starting Test for DPM+OSL..")
        working_dir = DATA_DIR + "DPM_OSL/"
        prepare_base_model(working_dir)
        cost, prepared_rewards, prepared_word_propensities, reweigh_sum = \
            train(saveto="%smodel.npz" % working_dir,
                  reload_=True,
                  shuffle_each_epoch=False,
                  datasets=DATA_SETS,
                  dictionaries=DICTIONARIES,
                  objective='CL',
                  cl_deterministic=True,
                  cl_log=LOG_PREFIX + ".json",
                  cl_reweigh=True,
                  unittest=True)

        true_cost = -0.8526537389124744
        self.assertAlmostEqual(true_cost, cost)
        true_prepared_rewards = numpy.array([1., 0.8222672, 1., 0.8274377, 0.7813821, 0.7813821, 0.6673543, 1., 0.6093617, 1.])
        numpy.testing.assert_almost_equal(true_prepared_rewards, prepared_rewards)
        true_prepared_word_propensities = numpy.zeros(shape=(22, 10))
        numpy.testing.assert_almost_equal(true_prepared_word_propensities, prepared_word_propensities)
        true_reweigh_sum = 0.826496987098
        self.assertAlmostEqual(true_reweigh_sum, reweigh_sum)
        shutil.rmtree(working_dir)
        logging.info("Finished Test for DPM+OSL..")
        return 0

    def test_IPS(self):
        """
        Verifies that the correct cost is calculated for IPS and sequence level rewards.
        :return: 0 on success
        """
        logging.info("Starting Test for IPS..")
        working_dir = DATA_DIR + "IPS/"
        prepare_base_model(working_dir)
        cost, prepared_rewards, prepared_word_propensities, reweigh_sum = \
            train(saveto="%smodel.npz" % working_dir,
                  reload_=True,
                  shuffle_each_epoch=False,
                  datasets=DATA_SETS,
                  dictionaries=DICTIONARIES,
                  objective='CL',
                  cl_deterministic=False,
                  cl_log=LOG_PREFIX + ".json",
                  unittest=True)

        true_cost = -0.8489172076113973
        self.assertAlmostEqual(true_cost, cost)
        true_prepared_rewards = numpy.array([1., 0.8222672, 1., 0.8274377, 0.7813821, 0.7813821, 0.6673543, 1., 0.6093617, 1.])
        numpy.testing.assert_almost_equal(true_prepared_rewards, prepared_rewards)
        true_prepared_word_propensities = [[ 1.07288418e-06, 2.14576951e-06, 2.38418608e-07, 0., 0., 0., 1.19209297e-07, 0., 0., 4.41075344e-06], [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [ 9.17915742e-06, 0., 4.33240789e-04, 1.16825786e-05, 0., 0., 0., 0., 0., 0.], [ 1.07288418e-06, 0., 2.38418608e-07, 0., 0., 0., 0., 0., 0., 0.], [ 0., 0., 0., 0., 2.38418608e-07, 1.19209297e-07, 0., 0., 3.57627933e-07, 0.], [ 0., 0., 0., 0., 0., 0., 9.41757822e-06, 7.15255993e-07, 0., 0.], [ 1.90735045e-06, 7.15255993e-07, 4.76837272e-07, 3.57627933e-07, 0., 0., 0., 0., 0., 0.], [ 0., 0., 0., 0., 8.77356514e-02, 1.19209297e-07, 1.11982840e-02, 0., 3.45712915e-05, 0.], [ 0., 0., 0., 0., 2.63984707e-01, 5.87784793e-01, 6.93070980e-03, 2.38418608e-07, 1.99701978e-03, 2.14576951e-06], [ 7.35016515e-04, 0., 0., 0., 0., 0., 0., 0., 0., 0.], [ 1.13374461e-04, 6.50148525e-01, 2.08618433e-05, 1.68086511e-05, 0., 0., 0., 0., 1.27681300e-04, 0.], [ 0., 0., 3.09762325e-01, 2.37591715e-01, 1.19209297e-07, 0., 0., 0., 0., 0.], [ 0., 0., 1.34707404e-05, 0., 0., 0., 0., 0., 0., 0.], [ 1.54972196e-06, 0., 1.31130304e-06, 0., 0., 0., 0., 0., 0., 0.], [ 0., 1.19209297e-07, 0., 0., 0., 0., 0., 0., 0., 0.], [ 7.98734138e-05, 3.50157816e-03, 0., 0., 0., 0., 0., 0., 0., 0.], [ 0., 5.96046625e-07, 0., 0., 0., 0., 0., 0., 0., 0.], [ 1.35233010e-03, 1.14441573e-05, 0., 0., 0., 0., 0., 0., 0., 0.], [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [ 0., 2.74181742e-06, 0., 0., 0., 0., 0., 0., 0., 0.], [ 0., 2.63638265e-02, 0., 0., 0., 0., 0., 0., 0., 0.], [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
        numpy.testing.assert_almost_equal(true_prepared_word_propensities, prepared_word_propensities)
        true_reweigh_sum = 0.0
        self.assertAlmostEqual(true_reweigh_sum, reweigh_sum)
        shutil.rmtree(working_dir)
        logging.info("Finished Test for IPS..")
        return 0

    def test_DPM_T(self):
        """
        Verifies that the correct cost is calculated for DPM+T.
        :return: 0 on success
        """
        logging.info("Starting Test for DPM+T..")
        working_dir = DATA_DIR + "DPM_T/"
        prepare_base_model(working_dir)
        cost, prepared_rewards, prepared_word_propensities, reweigh_sum = \
            train(saveto="%smodel.npz" % working_dir,
                  reload_=True,
                  shuffle_each_epoch=False,
                  datasets=DATA_SETS,
                  dictionaries=DICTIONARIES,
                  objective='CL',
                  cl_deterministic=True,
                  cl_log=LOG_PREFIX + ".json",
                  cl_external_reward=WORD_REWARD,
                  cl_word_rewards=True,
                  unittest=True)

        true_cost = -0.9641428801812324
        self.assertAlmostEqual(true_cost, cost)
        true_prepared_rewards = numpy.array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 0., 1., 0., 1.], [1., 1., 1., 1., 0., 0., 0., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 0., 1., 1., 1., 1., 1., 1., 0., 1.], [1., 1., 1., 0., 1., 1., 0., 0., 0., 0.], [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.], [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.], [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.], [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.], [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        numpy.testing.assert_almost_equal(true_prepared_rewards, prepared_rewards)
        true_prepared_word_propensities = numpy.zeros(shape=(22, 10))
        numpy.testing.assert_almost_equal(true_prepared_word_propensities, prepared_word_propensities)
        true_reweigh_sum = 0.0
        numpy.testing.assert_almost_equal(true_reweigh_sum, reweigh_sum)
        shutil.rmtree(working_dir)
        logging.info("Finished Test for DPM+T..")
        return 0

    def test_DPM_T_OSL(self):
        """
        Verifies that the correct cost is calculated for DPM+T+OSL
        :return: 0 on success
        """
        logging.info("Starting Test for DPM+T+OSL..")
        working_dir = DATA_DIR + "DPM_T_OSL/"
        prepare_base_model(working_dir)
        cost, prepared_rewards, prepared_word_propensities, reweigh_sum = \
            train(saveto="%smodel.npz" % working_dir,
                  reload_=True,
                  shuffle_each_epoch=False,
                  datasets=DATA_SETS,
                  dictionaries=DICTIONARIES,
                  objective='CL',
                  cl_deterministic=True,
                  cl_log=LOG_PREFIX + ".json",
                  cl_external_reward=WORD_REWARD,
                  cl_reweigh=True,
                  cl_word_rewards=True,
                  unittest=True)

        true_cost = -1.1665413125898798
        self.assertAlmostEqual(true_cost, cost)
        true_prepared_rewards = numpy.array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 0., 1., 0., 1.], [1., 1., 1., 1., 0., 0., 0., 1., 1., 1.], [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.], [1., 0., 1., 1., 1., 1., 1., 1., 0., 1.], [1., 1., 1., 0., 1., 1., 0., 0., 0., 0.], [1., 1., 1., 1., 1., 1., 0., 0., 0., 0.], [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.], [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.], [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.], [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
        numpy.testing.assert_almost_equal(true_prepared_rewards, prepared_rewards)
        true_prepared_word_propensities = numpy.zeros(shape=(22, 10))
        numpy.testing.assert_almost_equal(true_prepared_word_propensities, prepared_word_propensities)
        true_reweigh_sum = 0.826496987098
        numpy.testing.assert_almost_equal(true_reweigh_sum, reweigh_sum)
        shutil.rmtree(working_dir)
        logging.info("Finished Test for DPM+T+OSL..")
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
