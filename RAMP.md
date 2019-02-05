 RAMP LOSS OBJECTIVES
-----------------------
This repository implements the objectives RAMP, RAMP-T, RAMP1 and RAMP2 from [Jehl et al., 2019](TODO).

SCRIPT CALLS
------------
The RAMP objective can be called with the following arguments:

```
from nematus.nmt import train
train(saveto="path/to/dir",
        datasets=['/path/to/train/src/file'],
        valid_datasets=['/path/to/validation/src/file', '/path/to/validation/tgt/file'],
        dictionaries=['/path/to/src/vocabulary.json', '/path/to/tgt/vocabulary.json'],
        objective='RAMP',
        ramp_weak_gold='/path/to/train/gold_downstream/file',
        ramp_execution_script='/path/to/external/script/to/get/weak/feedback')
```

About the arguments:
 - datasets: For ramp, only the source file is specified as we assume that no gold targets are available for training
 - ramp_weak_gold: Here, if available, the weaker gold values can be specified (e.g. for semantic parsing, the gold answers)
 - ramp_execution_script: external script that might be required to obtain the weak feedback (e.g. for semantic parsing, it is a script that handles the database call to retrieve the answer for an output suggested by the system)

For RAMP+T, add:
```
    ramp_word_rewards=True
```

For RAMP1, add:
```
    ramp_top1_is_hope=True
```

For RAMP2, add:
```
    ramp_top1_is_fear=True
```

Additional arguments:
 - ramp_nbest: The number of n-best entries that should be retrieved and which we will normalise over
 - ramp_nloop: The number of n-best entries to search for a hope and a fear before skipping an example if either hope or fear are not found
 - ramp_normalization_alpha: Temperature parameter when normalizing the n-best list
 - ramp_reuse_previous: Caches previously found hope and fear, so they can be re-used if no hope/fear can be found later
 - use_memcache: Can be used to cache previous feedback obtained for model outputs, e.g. for semantic parsing it can cache answers for produced queries to save additional database calls (If memcache is not used, a simple python dictionary is used for caching instead)
 
NEW TASKS
------------
To implement a new task, the process of how hope and fear are selected needs to be adjusted, see nematus/nmt.py, lines 1905 - 1949
 
SEMANTIC PARSING: NLmaps
------------
The NLmaps corpus can be downloaded [here](https://www.cl.uni-heidelberg.de/statnlpgroup/nlmaps/).

Scripts to run the NLmaps semantic parsing task can be found in the subfolder scripts_ramp_parsing. There are:

 - ramp_exec_nlmapsv2.sh: point to this script for the argument --ramp_execution_script: It handles converting a linearised output of the system into a fully formed query and executes it against the OSM database
 - validate_nlmaps.sh: point to this script for the argument --external_validation_script: At validation time, calling this script will validate the current model on the NLmaps development data and returns the F1 score by comparing the resulting answers to gold answers.
 - validate_dev.sh: called by validate_nlmaps.sh
 - validate_test.sh: script that evaluates the nlmaps test set
 
Further NLmaps relevant scripts references in validate*.sh can be found [here](https://github.com/carhaas/scripts_nlmaps) and the repository for the OSM database that can handle NLmaps queries is [here](https://github.com/carhaas/overpass-nlmaps).

The validate_dev.sh file computes sequence level and answer level metrics on the dev set and save the model if it is better than the previous one.
The validate_test.sh file computes sequence level and answer level metrics on the test set.
(Both for NLmaps v2, split 2)

The MRT training has also been modified to work with the semantic parsing task. For MRT training, set the argument objective='MRT' while also suppling the arguments ramp_execution_script and ramp_weak_gold.