COUNTERFACTUAL LEARNING
-----------------------
This repo implements the objectives DPM, DPM+OSL, DPM+T and DPM+T+OSL from [Lawrence & Riezler, 2018](https://arxiv.org/abs/1805.01252).

LOG CREATION
------------
Given a base model, logs can be created with the following command:

```
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=gpu,on_unused_input=warn python nematus/translate.py \
     -m data_unittest/base_model/model.npz \
     -i data_unittest/toy.pre \
     -r data_unittest/toy.lin \
     -k 12 -p 1 --suppress-unk \
     -o log.tgt \
     -l log.json >toy.log 2>&1 &
```
 
 The log will be saved in ``log.json`` and each line will have a json array of the following format:
 
 - the source sentence
 - the most likely translation of the source sentence according to the base model
 - an array of word probability of the words in the translation
 - per-sentence BLEU reward
 - probability for the entire translation sequence
 
 The default reward for logs is per-sentence BLEU. But separate reward files can be created and handed over to the learner. In a separate reward file, each line should contain a reward and it will be matched to the same line in the json log.

LEARNING
--------
Once a log is created, any previous model can be loaded and counterfactual learning can commence. An example script is provided in [counterfactual_learning_example.py](https://github.com/carhaas/nematus/blob/master/scripts_counterfactual/counterfactual_learning_example.py).

For counterfactual learning, the following parameters have to be set:
```
objective='CL'
cl_log='log.json'
```

External rewards should be supplied using
```
cl_external_reward=PATH/TO/REWARDS
```

The parameters for the different objectives are:

DPM
```
cl_deterministic=True
```

DPM+OSL
```
cl_deterministic=True
cl_reweigh=True
cl_word_rewards=True
cl_external_reward=PATH/TO/WORD_REWARDS
```

DPM+T
```
cl_deterministic=True
```

DPM+T+OSL
```
cl_deterministic=True
cl_reweigh=True
cl_word_rewards=True
cl_external_reward=PATH/TO/WORD_REWARDS
```

NOTE
----
Note that unittests cannot be run without the correct model.npz which is too large to upload.
