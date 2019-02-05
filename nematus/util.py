'''
Utility functions
'''

import sys
import json
import cPickle as pkl

#json loads strings as unicode; we currently still work with Python 2 strings, and need conversion
def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())

def load_dict(filename):
    try:
        with open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with open(filename, 'rb') as f:
            return pkl.load(f)


def load_config(basename):
    try:
        with open('%s.json' % basename, 'rb') as f:
            return json.load(f)
    except:
        try:
            with open('%s.pkl' % basename, 'rb') as f:
                return pkl.load(f)
        except:
            sys.stderr.write('Error: config file {0}.json is missing\n'.format(basename))
            sys.exit(1)


def seqs2words(seq, inverse_target_dictionary, join=True):
    words = []
    for w in seq:
        if w == 0:
            break
        if w in inverse_target_dictionary:
            words.append(inverse_target_dictionary[w])
        else:
            words.append('UNK')
    return ' '.join(words) if join else words



def write_list_to_file(list, file_to_write):
    """ Iterates over the entries in a list and writes them to a file,
    one list entry corresponds to one line in the file

    :param list: the list to be written to a file
    :param file_to_write: the file to write to
    :return: 0 on success
    """
    with open(file_to_write, 'w') as f:
        for line in list:
            f.write(line.rstrip('\n')+"\n")
    return 0


def read_lines_in_list(file_to_read):
    """ Iterates over the lines in a file and adds the line to a list

    :param file_to_read: the location of the file to be read
    :return: a list where each entry corresponds to a line in the file
    """
    collect_list = []
    with open(file_to_read, 'r') as f:
        for line in f:
            collect_list.append(line.rstrip('\n'))
    return collect_list
