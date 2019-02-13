#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

from math import exp
from operator import mul
from collections import defaultdict

from scorer import Scorer
from reference import Reference

import logging

class SeqScorer(Scorer):
    """
    Scores SmoothedBleuReference objects.
    """

    def __init__(self, argument_string):
        """
        Initialises metric-specific parameters.
        """
        Scorer.__init__(self, argument_string)
        if not 'negative_value' in self._arguments.keys():
            self._arguments['negative_value'] = 0.0

    def set_reference(self, reference_tokens):
        """
        Sets the reference against hypotheses are scored.
        """
        self._reference = SeqScorerReference(
            reference_tokens,
            self._arguments['negative_value']
        )

class SeqScorerReference(Reference):
    """
    Smoothed sentence-level BLEU as as proposed by Lin and Och (2004).
    Implemented as described in (Chen and Cherry, 2014).
    """

    def __init__(self, reference_tokens, negative_value=0.0):
        """
        @param reference the reference translation that hypotheses shall be
                         scored against. Must be an iterable of tokens (any
                         type).
        """
        Reference.__init__(self, reference_tokens)
        self.negative_value = negative_value

    def score(self, hypothesis_tokens):
        """
        Scores @param hypothesis against this reference.

        @return the smoothed sentence-level BLEU score: 1.0 is best, 0.0 worst.
        """
        if hypothesis_tokens == self._reference_tokens:
            return 1.0
        else:
            return self.negative_value