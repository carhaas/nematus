import numpy
import os
import sys
import shutil

VOCAB_SIZE = 90000
SRC = "pre"
TGT = "lin"
NEMATUS="/PATH/TO/NEMATUS/"
DATA_DIR = "/PATH/TO/DATA/"
LOG = "/PATH/TO/LOG_FILE/" # has to be created first using some logging model, see COUNTERFACTUAL.md
FILENAME = os.path.basename(__file__)
FILENAME = FILENAME.replace(".py", "")
LOC = "%s/" % FILENAME
BASE_MODEL = "/PATH/TO/BASE_MODEL/" # model that should be improved, ideally this model should be identical to the model that created the log but they can differ

from nematus.nmt import train

if __name__ == '__main__':
    shutil.copytree(BASE_MODEL, LOC)
    validerr = train(saveto="%smodel.npz" % LOC,
                    reload_=True,
                    patience=50,
                    dim_word=1000,
                    dim=1024,
                    shuffle_each_epoch=True,
                    clip_c=1.,
                    decay_c=1.0e-8,
                    lrate=0.0001,
                    optimizer='adadelta',
                    maxlen=201,
                    batch_size=80,
                    valid_batch_size=80,
                    datasets=[DATA_DIR + '/PATH/TO/SOURCES_FROM_LOG/' + SRC, LOG + '.tgt'], # first: file with log source sentences, second: file with translations from the model that created the log
                    valid_datasets=[DATA_DIR + '/PATH/TO/VALIDATION_SOURCE/' + SRC, DATA_DIR + '/PATH/TO/VALIDATION_REFERENCE/' + TGT],
                    dictionaries=[DATA_DIR + '/PATH/TO/SOURCE_DICTIONARY_FROM_BASE_MODEL/' + SRC + '.json',DATA_DIR + '/PATH/TO/TARGET_DICTIONARY_FROM_BASE_MODEL/' + TGT + '.json'],
                    validFreq=100,
                    sampleFreq=100,
                    saveFreq=100,
                    dispFreq=100,
                    use_dropout=False,
                    dropout_embedding=0.0, # dropout for input embeddings (0: no dropout)
                    dropout_hidden=0.0, # dropout for hidden layers (0: no dropout)
                    dropout_source=0.0, # dropout source words (0: no dropout)
                    dropout_target=0.0, # dropout target words (0: no dropout)
                    overwrite=True,
                    objective='CL',
                    cl_deterministic=True,
                    cl_log=LOG + '.json',
                    cl_reweigh=True,
                    cl_word_rewards=True,
                    cl_external_reward=NEMATUS+'PATH/TO/WORD_REWARDS',
                    external_validation_script='./validate_nlmaps.sh %s' % LOC)
print validerr
