import json
import gzip

import shuffle
from util import load_dict

import numpy


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class ClLogIterator:
    """Counterfactual Log iterator."""

    def __init__(self, source, target,
                 source_dicts, target_dict,
                 log,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 use_factor=False,
                 maxibatch_size=20,
                 external_reward=None,
                 word_rewards=False):
        if shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            self.log_orig = log
            self.external_reward_ori = external_reward
            if external_reward:
                self.source, self.target, self.log, self.external_reward = shuffle.main([self.source_orig, self.target_orig, self.log_orig, self.external_reward_ori], temporary=True)
            else:
                self.external_reward = None
                self.source, self.target, self.log = shuffle.main([self.source_orig, self.target_orig, self.log_orig], temporary=True)
        else:
            self.source = fopen(source, 'r')
            self.target = fopen(target, 'r')
            self.log = fopen(log, 'r')
            if external_reward:
                self.external_reward = fopen(external_reward, 'r')
            else:
                self.external_reward = None
        self.source_dicts = []
        for source_dict in source_dicts:
            self.source_dicts.append(load_dict(source_dict))
        self.target_dict = load_dict(target_dict)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.skip_empty = skip_empty
        self.use_factor = use_factor
        self.word_rewards = word_rewards

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
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.log_buffer = []
        self.external_reward_buffer = []
        self.k = batch_size * maxibatch_size

        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])

    def reset(self):
        if self.shuffle:
            if self.external_reward:
                self.source, self.target, self.log, self.external_reward = shuffle.main([self.source_orig, self.target_orig, self.log_orig, self.external_reward_ori], temporary=True)
            else:
                self.source, self.target, self.log = shuffle.main([self.source_orig, self.target_orig, self.log_orig], temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)
            self.log.seek(0)
            if self.external_reward:
                self.external_reward.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        reward = []
        seq_prob = []
        word_probs = []
        target_logged = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'
        assert len(self.source_buffer) == len(self.log_buffer), 'Buffer size mismatch!'
        if self.external_reward:
            assert len(self.source_buffer) == len(self.external_reward_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for ss in self.source:
                ss = ss.split()
                tt = self.target.readline().split()
                ll = json.loads(self.log.next())
                if self.external_reward:
                    if self.word_rewards:
                        rr = json.loads(self.external_reward.readline().strip())
                    else:
                        rr = self.external_reward.readline().strip()

                if self.skip_empty and (len(ss) == 0 or len(tt) == 0):
                    continue
                if len(ss) > self.maxlen or len(tt) > self.maxlen:
                    continue

                self.source_buffer.append(ss)
                self.target_buffer.append(tt)
                self.log_buffer.append(ll)
                if self.external_reward:
                    self.external_reward_buffer.append(rr)
                if len(self.source_buffer) == self.k:
                    break

            if len(self.source_buffer) == 0 or len(self.target_buffer) == 0 or len(self.log_buffer) == 0:
                self.end_of_data = False
                self.reset()
                raise StopIteration

            # sort by target buffer
            if self.sort_by_length:
                tlen = numpy.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]
                _lbuf = [self.log_buffer[i] for i in tidx]
                if self.external_reward:
                    _rbuf = [self.external_reward_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf
                self.log_buffer = _lbuf
                if self.external_reward:
                    self.external_reward_buffer = _rbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()
                self.log_buffer.reverse()
                if self.external_reward:
                    self.external_reward_buffer.reverse()

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

                # read from target file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.target_dict[w] if w in self.target_dict else 1
                      for w in tt]

                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target else 1 for w in tt]

                # read from log file and convert to correct structure
                ll = self.log_buffer.pop()
                if self.external_reward:
                    rr = self.external_reward_buffer.pop()

                current_y_log = ll[1].split()
                current_y_log = [self.target_dict[w] if w in self.target_dict else 1
                      for w in current_y_log]

                if self.n_words_target > 0:
                    current_y_log = [w if w < self.n_words_target else 1 for w in current_y_log]

                if self.external_reward:
                    current_reward = rr
                else:
                    current_reward = ll[3]

                current_seq_prob = ll[4]
                current_word_probs = ll[2]

                source.append(ss)
                target.append(tt)
                reward.append(current_reward)
                seq_prob.append(current_seq_prob)
                word_probs.append(current_word_probs)
                target_logged.append(current_y_log)

                if len(source) >= self.batch_size or \
                                len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        return source, zip(target, reward, seq_prob, target_logged, word_probs)
