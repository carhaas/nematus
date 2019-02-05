#!/bin/sh

#model prefix
prefix=$1

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/path/to/nematus

# see ( https://github.com/carhaas/scripts_nlmaps ) to find the relevant scripts
# changes a linearise MRL into a fully formed one that can be executed against the database
python $nematus/scripts_nlmaps/functionalise.py -i ${prefix}.ramp.lin -o ${prefix}.ramp.mrl 2>${prefix}.err >${prefix}.out

# see ( https://github.com/carhaas/overpass-nlmaps )
# call to thhe database to obtain an answer
$nematus/osm/osm3s_v0.7.51_production/query_db -d $DB_DIR -a ${prefix}.ramp.answer -f ${prefix}.ramp.mrl 2>err >out

