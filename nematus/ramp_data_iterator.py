import numpy

import gzip

import shuffle
from util import load_dict

import logging

def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class RampTextIterator:
    """Ramp iterator, returns source and answer."""

    def __init__(self, source,
                 source_dicts, target_dict,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 use_factor=False,
                 maxibatch_size=20,
                 ramp_weak_gold=None):
        if shuffle_each_epoch:
            self.source_orig = source
            self.ramp_weak_gold_orig = ramp_weak_gold
            self.source, self.ramp_weak_gold = shuffle.main([self.source_orig, self.ramp_weak_gold_orig], temporary=True)
        else:
            self.source = fopen(source, 'r')
            self.ramp_weak_gold = fopen(ramp_weak_gold, 'r')
        self.source_dicts = []
        for source_dict in source_dicts:
            self.source_dicts.append(load_dict(source_dict))
        self.target_dict = load_dict(target_dict)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty
        self.use_factor = use_factor

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        if self.n_words_source > 0:
            for d in self.source_dicts:
                for key, idx in d.items():
                    if idx >= self.n_words_source:
                        del d[key]

        if self.n_words_target > 0:
            for key, idx in self.target_dict.items():
                if idx >= self.n_words_target:
                    del self.target_dict[key]

        self.shuffle = shuffle_each_epoch

        self.source_buffer = []
        self.ramp_weak_gold_buffer = []
        self.k = batch_size * maxibatch_size

        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])

    def reset(self):
        if self.shuffle:
            self.source, self.ramp_weak_gold = shuffle.main([self.source_orig, self.ramp_weak_gold_orig], temporary=True)
        else:
            self.source.seek(0)
            self.ramp_weak_gold.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        ramp_weak_gold = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.ramp_weak_gold_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for ss in self.source:
                ss = ss.split()
                rr = self.ramp_weak_gold.readline().strip()

                if self.skip_empty and (len(ss) == 0):
                    continue
                if len(ss) > self.maxlen:
                    continue

                self.source_buffer.append(ss)
                self.ramp_weak_gold_buffer.append(rr)
                if len(self.source_buffer) == self.k:
                    break

            if len(self.source_buffer) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration

            self.source_buffer.reverse()
            self.ramp_weak_gold_buffer.reverse()

        try:
            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                tmp = []
                for w in ss:
                    if self.use_factor:
                        w = [self.source_dicts[i][f] if f in self.source_dicts[i] else 1 for (i, f) in
                             enumerate(w.split('|'))]
                    else:
                        w = [self.source_dicts[0][w] if w in self.source_dicts[0] else 1]
                    tmp.append(w)
                ss = tmp

                # read from source file and map to word index
                test = self.ramp_weak_gold_buffer.pop()
                ramp_weak_gold.append(test)

                source.append(ss)

                if len(source) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True
        return source, ramp_weak_gold
