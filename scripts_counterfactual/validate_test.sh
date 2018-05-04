#!/bin/sh

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/home/mitarb/haas1/nematus

# theano device, in case you do not want to compute on gpu, change it to cpu
device=gpu

# model prefix
prefix=$1/model.npz

dev=/PATH/TO/TEST_SOURCE
ref=/PATH/TO/TEST_REFERENCE
answer=/PATH/TO/TEST_ANSWERS #for semantic parsing on NLmaps

# path to scripts/multi-bleu
BLEU_SCRIPT=/PATH/TO/BLEU_SCRIPT
SEQEVAL_SCRIPT=/PATH/TO/SEQEVAL_SCRIPT # see https://github.com/carhaas/scripts_nlmaps
# required for validate NLmaps against answers
F1_SCRIPT=/PATH/TO/F1_SCRIPT # see https://github.com/carhaas/scripts_nlmaps
FUNCTIONALISE_SCRIPT=/PATH/TO/FUNCTIONALISE_SCRIPT # see https://github.com/carhaas/scripts_nlmaps
QUERY_DB=/PATH/TO/QUERY_DB # see https://github.com/carhaas/overpass-nlmaps
DB_DIR=/PATH/TO/OSM_DB # see https://github.com/carhaas/overpass-nlmaps

# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m ${prefix}.dev.npz \
     -i $test \
     -o ${prefix}.output.test \
     -k 12 -n -p 1


## get BLEU
BEST=`cat ${prefix}_best_test_bleu || echo 0`
$BLEU_SCRIPT $ref < ${prefix}.output.test >> ${prefix}_test_bleu_scores
BLEU=`$nematus/scripts/multi-bleu.perl $ref < ${prefix}.output.test | cut -f 3 -d ' ' | cut -f 1 -d ','`
BETTER=`echo "$BLEU > $BEST" | bc`
echo $BETTER

## get SEQEVAL
SEQEVAL_BEST=`cat ${prefix}_best_test_seqeval || echo 0`
python $SEQEVAL_SCRIPT -i ${prefix}.output.test -g $ref >> ${prefix}_test_eval_scores
SEQEVAL=`python $nematus/scripts_nlmaps/seq_eval.py -i ${prefix}.output.test -g $ref`
BETTER_SEQ=`echo "$SEQEVAL > $SEQEVAL_BEST" | bc`
echo $BETTER

## get F1
save_dir=$1/test_files
mkdir $save_dir
cp ${prefix}.output.test $save_dir/output.test.lin.$2
BEST=`cat ${prefix}_best_test_f1 || echo 0`
python $FUNCTIONALISE_SCRIPT -i ${prefix}.output.test -o ${prefix}.output.test.mrl 2>err >out
cp ${prefix}.output.test.mrl $save_dir/output.test.mrl.$2

$QUERY_DB -d $DB_DIR -a ${prefix}.output.test.answers -f ${prefix}.output.test.mrl 2>err >out
cp ${prefix}.output.test.answers $save_dir/output.test.answers.$2
python $F1_SCRIPT -i ${prefix}.output.test.answers -g $answer >>${prefix}_test_f1_scores
F1=`python $nematus/scripts_nlmaps/eval.py -i ${prefix}.output.test.answers -g $answer`
BETTER=`echo "$F1 > $BEST" | bc`

echo "TEST F1 = $F1"
