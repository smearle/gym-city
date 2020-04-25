install:
	cd gym_city/envs/micropolis/MicropolisCore; make; sudo make install

train:
	python3 train_teacher.py --eval-interval 2000 --vis-interval 2000000 --model FractalNet --drop --n-recs 3 --load --load-dir '/home/sme/gym-city/trained_models/a2c_FractalNet_drop/MicropolisEnv-v0_w16_200s__noXt3_alpgmm_weighted' --poet True --num-frames 2000000000 --num-proc 125

teach_GoL:
	python3 main_teacher.py --env-name GoLMultiEnv-v0 --algo a2c --experiment test --map-width 16 --model FullyConv --num-frames 100000000 --eval-interval 1000 --num-processes 500 --render
