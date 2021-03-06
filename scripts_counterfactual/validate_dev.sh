#!/bin/sh

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/PATH/TO/NEMATUS

# theano device, in case you do not want to compute on gpu, change it to cpu
device=gpu

# model prefix
prefix=$1/model.npz

dev=/PATH/TO/VALIDATION_SOURCE
ref=/PATH/TO/VALIDATION_REFERENCE
answer=/PATH/TO/VALIDATION_ANSWERS #for semantic parsing on NLmaps

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
     -i $dev \
     -o ${prefix}.output.dev \
     -k 12 -n -p 1


## get BLEU
BEST=`cat ${prefix}_best_bleu || echo 0`
$BLEU_SCRIPT $ref < ${prefix}.output.dev >> ${prefix}_bleu_scores
BLEU=`$nematus/scripts/multi-bleu.perl $ref < ${prefix}.output.dev | cut -f 3 -d ' ' | cut -f 1 -d ','`
BETTER=`echo "$BLEU > $BEST" | bc`

# save model with highest BLEU
if [ "$BETTER" = "1" ]; then
  echo "new best; saving"
  echo $BLEU > ${prefix}_best_bleu
  cp ${prefix}.dev.npz ${prefix}.best_bleu.npz
  cp ${prefix}.dev.npz.json ${prefix}.best_bleu.npz.json
  cp ${prefix}.dev.progress.json ${prefix}.best_bleu.progress.json
  cp ${prefix}.dev.gradinfo.npz ${prefix}.best_bleu.gradinfo.npz
fi

## get SEQEVAL
SEQEVAL_BEST=`cat ${prefix}_best_seqeval || echo 0`
python $SEQEVAL_SCRIPT -i ${prefix}.output.dev -g $ref >> ${prefix}_eval_scores
SEQEVAL=`python $nematus/scripts_nlmaps/seq_eval.py -i ${prefix}.output.dev -g $ref`
BETTER_SEQ=`echo "$SEQEVAL > $SEQEVAL_BEST" | bc`

# save model with highest SEQEVAL
if [ "$BETTER_SEQ" = "1" ]; then
  echo "new best; saving"
  echo $SEQEVAL > ${prefix}_best_seqeval
  cp ${prefix}.dev.npz ${prefix}.best_seqeval.npz
  cp ${prefix}.dev.npz.json ${prefix}.best_seqeval.npz.json
  cp ${prefix}.dev.progress.json ${prefix}.best_seqeval.progress.json
  cp ${prefix}.dev.gradinfo.npz ${prefix}.best_seqeval.gradinfo.npz
fi

## get F1
save_dir=$1/dev_files
mkdir $save_dir
cp ${prefix}.output.dev $save_dir/output.dev.lin.$2
BEST=`cat ${prefix}_best_f1 || echo 0`
python $FUNCTIONALISE_SCRIPT -i ${prefix}.output.dev -o ${prefix}.output.dev.mrl 2>err >out
cp ${prefix}.output.dev.mrl $save_dir/output.dev.mrl.$2

$QUERY_DB -d $DB_DIR -a ${prefix}.output.dev.answers -f ${prefix}.output.dev.mrl 2>err >out
cp ${prefix}.output.dev.answers $save_dir/output.dev.answers.$2
python $F1_SCRIPT -i ${prefix}.output.dev.answers -g $answer >>${prefix}_f1_scores
F1=`python $nematus/scripts_nlmaps/eval.py -i ${prefix}.output.dev.answers -g $answer`
BETTER=`echo "$F1 > $BEST" | bc`

echo "DEV F1 = $F1"

 # save model with highest F1
if [ "$BETTER" = "1" ]; then
  echo "new best; saving"
  echo $F1 > ${prefix}_best_f1
  cp ${prefix}.output.dev.mrl ${prefix}.output.dev.mrl.best_f1
  cp ${prefix}.output.dev.answers ${prefix}.output.dev.answers.best_f1
  cp ${prefix}.output.dev.answers.sigf ${prefix}.output.dev.answers.sigf.best_f1
  cp ${prefix}.output.dev.answers.eval ${prefix}.output.dev.answers.eval.best_f1
  cp ${prefix}.dev.npz ${prefix}.best_f1.npz
  cp ${prefix}.dev.npz.json ${prefix}.best_f1.npz.json
  cp ${prefix}.dev.progress.json ${prefix}.best_f1.progress.json
  cp ${prefix}.dev.gradinfo.npz ${prefix}.best_f1.gradinfo.npz
 fi
