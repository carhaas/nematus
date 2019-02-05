#!/bin/sh

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/path/to/nematus
# assumes that $nematus/scripts_nlmaps contains the repo of: https://github.com/carhaas/scripts_nlmaps
# and $nematus/overpass-nlmaps contains the repo of: https://github.com/carhaas/overpass-nlmaps

# theano device, in case you do not want to compute on gpu, change it to cpu
device=gpu

#model prefix
prefix=$1/model.npz

test=nlmaps/nlmaps.v2.log.test.pre
ref=nlmaps/nlmaps.v2.log.test.lin
answer=nlmaps/nlmaps.v2.log.test.gold_may16

# decode
THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device=$device,on_unused_input=warn python $nematus/nematus/translate.py \
     -m ${prefix}.dev.npz \
     -i $test \
     -o ${prefix}.output.test \
     -k 12 -n -p 1

## get SEQEVAL
SEQEVAL_BEST=`cat ${prefix}_best_test_seqeval || echo 0`
python $nematus/scripts_nlmaps/seq_eval.py -i ${prefix}.output.test -g $ref >> ${prefix}_test_eval_scores
SEQEVAL=`python $nematus/scripts_nlmaps/seq_eval.py -i ${prefix}.output.test -g $ref`
BETTER_SEQ=`echo "$SEQEVAL > $SEQEVAL_BEST" | bc`
echo $BETTER
echo "SEQEVAL = $SEQEVAL"

## get F1
save_dir=$1/test_files
mkdir $save_dir
cp ${prefix}.output.test $save_dir/output.test.lin.$2
BEST=`cat ${prefix}_best_test_f1 || echo 0`
python $nematus/scripts_nlmaps/functionalise.py -i ${prefix}.output.test -o ${prefix}.output.test.mrl 2>err >out
cp ${prefix}.output.test.mrl $save_dir/output.test.mrl.$2

$nematus/overpass-nlmaps/query_db -d $DB_DIR -a ${prefix}.output.test.answers -f ${prefix}.output.test.mrl 2>err >out
cp ${prefix}.output.test.answers $save_dir/output.test.answers.$2
python $nematus/scripts_nlmaps/eval.py -i ${prefix}.output.test.answers -g $answer >>${prefix}_test_f1_scores
F1=`python $nematus/scripts_nlmaps/eval.py -i ${prefix}.output.test.answers -g $answer`
BETTER=`echo "$F1 > $BEST" | bc`

echo "F1 = $F1"

 # save model with highest F1
if [ "$BETTER" = "1" ]; then
  echo "new best; saving"
  echo $F1 > ${prefix}_best_test_f1
  cp ${prefix}.output.test.mrl ${prefix}.output.test.mrl.best_f1
  cp ${prefix}.output.test.answers ${prefix}.output.test.answers.best_f1
  cp ${prefix}.output.test.answers.sigf ${prefix}.output.test.answers.sigf.best_f1
  cp ${prefix}.output.test.answers.eval ${prefix}.output.test.answers.eval.best_f1
  cp ${prefix}.dev.npz ${prefix}.test_best_f1.npz
  cp ${prefix}.dev.npz.json ${prefix}.test_best_f1.npz.json
  cp ${prefix}.dev.progress.json ${prefix}.test_best_f1.progress.json
  cp ${prefix}.dev.gradinfo.npz ${prefix}.test_best_f1.gradinfo.npz
 fi
