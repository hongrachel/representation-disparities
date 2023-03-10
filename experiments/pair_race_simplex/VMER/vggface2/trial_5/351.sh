python3 main.py --dataset VMER --usr-config vmer --method vggface2 --experiment-name pair_race_simplex --simplex-max 600 --simplex-point-id 25 --num-classes 600 --trial-num 5 --train
python3 main.py --dataset VMER --usr-config vmer --test-race African --method vggface2 --experiment-name pair_race_simplex --simplex-max 600 --simplex-point-id 25 --num-classes 600 --trial-num 5 --test
python3 main.py --dataset VMER --usr-config vmer --test-race Asian --method vggface2 --experiment-name pair_race_simplex --simplex-max 600 --simplex-point-id 25 --num-classes 600 --trial-num 5 --test
python3 main.py --dataset VMER --usr-config vmer --test-race Caucasian --method vggface2 --experiment-name pair_race_simplex --simplex-max 600 --simplex-point-id 25 --num-classes 600 --trial-num 5 --test
python3 main.py --dataset VMER --usr-config vmer --test-race Indian --method vggface2 --experiment-name pair_race_simplex --simplex-max 600 --simplex-point-id 25 --num-classes 600 --trial-num 5 --test
