python3 main.py --dataset RFW --usr-config default --method vggface2 --experiment-name pair_race_simplex --simplex-max 4500 --simplex-point-id 50 --num-classes 4500 --trial-num 5 --train
python3 main.py --dataset RFW --usr-config default --test-race African --method vggface2 --experiment-name pair_race_simplex --simplex-max 4500 --simplex-point-id 50 --num-classes 4500 --trial-num 5 --test
python3 main.py --dataset RFW --usr-config default --test-race Asian --method vggface2 --experiment-name pair_race_simplex --simplex-max 4500 --simplex-point-id 50 --num-classes 4500 --trial-num 5 --test
python3 main.py --dataset RFW --usr-config default --test-race Caucasian --method vggface2 --experiment-name pair_race_simplex --simplex-max 4500 --simplex-point-id 50 --num-classes 4500 --trial-num 5 --test
python3 main.py --dataset RFW --usr-config default --test-race Indian --method vggface2 --experiment-name pair_race_simplex --simplex-max 4500 --simplex-point-id 50 --num-classes 4500 --trial-num 5 --test
