#!/bin/bash

USER="rfw"
DATASET="RFW"
EXPERIMENT="1_to_2_increase_race_simplex_face_ratio"
# METHOD="vggface2"
TEST_RACES=("African" "Asian" "Caucasian" "Indian")
SIMPLEX_MAXES=`seq 2000 200 5000`
SIMPLEX_POINT_IDS=`seq 0 15`
METHODS=("vggface2" "arcface" "centerloss" "sphereface")

base="experiments/"
sweep="1_to_2_increase_race_simplex/"
dir=${base}${sweep}

for method in ${METHODS[@]}; do
    count=0
    out=${dir}/onetrial_face_ratios/${method}/
    for i in ${SIMPLEX_POINT_IDS[@]}; do
        for j in ${SIMPLEX_MAXES[@]}; do
            for k in `seq 0 0`; do
                line="python3 main.py --dataset "${DATASET}" --usr-config "${USER}" --method "${method}" --experiment-name "${EXPERIMENT}" --simplex-max "${j}" --simplex-point-id "${i}" --num-classes "${j}" --trial-num "${k}" --train"
                echo $line >> $out$count.sh

                for l in ${TEST_RACES[@]}; do
                    testline="python3 main.py --dataset "${DATASET}" --usr-config "${USER}" --test-race "${l}" --method "${method}" --experiment-name "${EXPERIMENT}" --simplex-max "${j}" --simplex-point-id "${i}" --num-classes "${j}" --trial-num "${k}" --test"
                    echo $testline >> $out$count.sh
                done
            done
            let "count++"
        done
    done

    # for i in `seq 0 3`; do
    #     for k in `seq 1 5`; do
    #         line="python3 main.py --dataset "${DATASET}" --usr-config "${USER}" --method "${method}" --experiment-name "${EXPERIMENT}" --simplex-max 2000 --simplex-point-id "${i}" --num-classes 2000 --trial-num "${k}" --train"
    #         echo $line >> $out$count.sh

    #         for l in ${TEST_RACES[@]}; do
    #             testline="python3 main.py --dataset "${DATASET}" --usr-config "${USER}" --test-race "${l}" --method "${method}" --experiment-name "${EXPERIMENT}" --simplex-max 2000 --simplex-point-id "${i}" --num-classes 2000 --trial-num "${k}" --test"
    #             echo $testline >> $out$count.sh
    #         done
    #     done
    #     let "count++"
    # done
done