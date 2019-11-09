install:
	cd gym_micropolis/envs/micropolis/MicropolisCore; make; sudo make install

teach_GoL:
	python3 main_teacher.py --env-name GoLMultiEnv-v0 --algo a2c --experiment test --num-frames 100000000 --eval-interval 1000 --num-processes 300 --render
