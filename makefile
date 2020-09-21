install:
	cd gym_city/envs/micropolis/MicropolisCore; make; sudo make install

train:
	python3 train_teacher.py --eval-interval 2000 --vis-interval 2000000 --model FractalNet --drop --n-recs 3 --load --load-dir '/home/sme/gym-city/trained_models/a2c_FractalNet_drop/MicropolisEnv-v0_w16_200s__noXt3_alpgmm_weighted' --poet True --num-frames 2000000000 --num-proc 125

train_dual:
	python train_dual.py --experiment train_dual_test --num-proc 2 --env-name zeldaplay_wide_v0 --overwrite --map-width 8 --render

######### Micropolis #########

MP_res_FC:
	python3 enjoy.py --load-dir '/home/sme/gym-city/trained_models/a2c_FullyConv/MicropolisEnv-v0_w16_200s_resOnlyThic/MicropolisEnv-v0.tar'   --poet True --map-w 16  --random-b  True --random-t True

MP_res_FC_w32:
	python3 enjoy.py --load-dir '/home/sme/gym-city/trained_models/a2c_FullyConv/MicropolisEnv-v0_w16_200s_resOnlyThic'   --poet True --map-w 32 --random-t

condos:
	python3 enjoy.py --load-dir '/home/sme/gym-city/trained_models/a2c_FullyConv/MicropolisEnv-v0_w16_200s_resOnlyThic'  --map-width 64 --render --model FullyConv --poet True --random-t

###

MP_SC_w16:
	python3 enjoy.py --load-dir  '/home/sme/gym-city/trained_models/a2c_FullyConv_w16/MicropolisEnv-v0_MP0' --map-w 16 --val-kern 3

MP_SC_w32:
	python3 enjoy.py --load-dir  '/home/sme/gym-city/trained_models/a2c_FullyConv_w16/MicropolisEnv-v0_MP0' --map-w 32 --val-kern 3 --random-b


bad_traffic:
	python3 enjoy.py --load-dir '/home/sme/gym-city/trained_models/a2c_FullyConv/MicropolisEnv-v0_w16_300s_trafficVec' --random-t --non-det

nice_mix:
	python3 enjoy.py --load-dir '/home/sme/gym-city/trained_models/a2c_FractalNet-5recs_intra_inter_drop/MicropolisEnv-v0_w16_200s_MP0' --n-chan 32 --active-col 0 --map-width 20

nice_mix_w32:
	python3 enjoy.py --load-dir '/home/sme/gym-city/trained_models/a2c_FractalNet-5recs_intra_inter_drop/MicropolisEnv-v0_w16_200s_MP0' --n-chan 32 --active-col 0 --map-width 32


######### Game of Life #########

GoL_SC:
	python3 enjoy.py --load-dir '/home/sme/gym-city/trained_models/a2c_FullyConv/GameOfLifeEnv-v0_w16_200s_MP_0' --n-chan 32 --val-kern 2 --model FullyConv --map-width 16 --render --max-step 300 --n-recs 5

GoL_SC_w32:
	python3 enjoy.py --load-dir '/home/sme/gym-city/trained_models/a2c_FullyConv/GameOfLifeEnv-v0_w16_200s_MP_0' --n-chan 32  --val-kern 2 --model FullyConv --map-width 16 --render --map-width 32 --max-step 700

GoL_SC_w64:
	python3 enjoy.py --load-dir '/home/sme/gym-city/trained_models/a2c_FullyConv/GameOfLifeEnv-v0_w16_200s_MP_0' --n-chan 32 --val-kern 2 --model FullyConv --map-width 16 --render --map-width 64 --max-step 2000

big_life:
	python3 enjoy.py --load-dir '/home/sme/gym-city/trained_models/a2c_FullyConv/GameOfLifeEnv-v0_w16_200s_MP_0' --n-chan 32 --val-kern 2 --model FullyConv --render --map-width 128 --max-step 5000 --prob-life 

######## Power Puzzle ########

powerless:
	python3 enjoy.py --load-dir '/home/sme/gym-city/trained_models/a2c_FractalNet-5recs_intra_drop/MicropolisEnv-v0_w16_200s_PP1' --render --n-chan 32 --map-width 16

