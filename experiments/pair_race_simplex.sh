#!/bin/bash

USER="vmer"
DATASET="VMER"
EXPERIMENT="pair_race_simplex"
# METHOD="vggface2"
TEST_RACES=("African" "Asian" "Caucasian" "Indian")
SIMPLEX_MAXES=`seq 50 50 800`
METHODS=("vggface2")

# 500_to_10000 j/500 * 6 -2 if j/500 <= 10 else (21-j/500) * 6
# 50_to_100 j/50 * 6 - 2 if j/50 <= 8 else (17 - j/50) * 6

base="experiments/"
sweep="pair_race_simplex/"
dir=${base}${sweep}${DATASET}

for method in ${METHODS[@]}; do
    for k in `seq 1 5`; do
        out=${dir}/${method}/trial_${k}/
        rm -r $out && mkdir $out
        count=0
        for j in ${SIMPLEX_MAXES[@]}; do
            num_points=0
            if [ $(( j/50 )) -le 8 ]
            then
                num_points="$((6 * j/50 - 2))"
            else
                num_points="$((6 * (17- j/50)))"
            fi
            last="$((num_points-1))"
            for i in `seq 0 $last`; do
                line="python3 main.py --dataset "${DATASET}" --usr-config "${USER}" --method "${method}" --experiment-name "${EXPERIMENT}" --simplex-max "${j}" --simplex-point-id "${i}" --num-classes "${j}" --trial-num "${k}" --train"
                echo $line >> $out$count.sh

                for l in ${TEST_RACES[@]}; do
                    testline="python3 main.py --dataset "${DATASET}" --usr-config "${USER}" --test-race "${l}" --method "${method}" --experiment-name "${EXPERIMENT}" --simplex-max "${j}" --simplex-point-id "${i}" --num-classes "${j}" --trial-num "${k}" --test"
                    echo $testline >> $out$count.sh
                done
                let "count++"
            done
        done
    done
done