install:
	cd gym_city/envs/micropolis/MicropolisCore; make; make install

clean:
	cd gym_city/envs/micropolis/MicropolisCore; make clean

######### Micropolis #########

MP_res_FC:
	python3 enjoy.py --load-dir '/home/sme/gym-city/trained_models/a2c_FullyConv/MicropolisEnv-v0_w16_200s_resOnlyThic'   --poet --map-w 16  --random-b  --random-t

MP_res_FC_w32:
	python3 enjoy.py --load-dir '/home/sme/gym-city/trained_models/a2c_FullyConv/MicropolisEnv-v0_w16_200s_resOnlyThic'   --poet --map-w 32 --random-t

condos:
	python3 enjoy.py --load-dir '/home/sme/gym-city/trained_models/a2c_FullyConv/MicropolisEnv-v0_w16_200s_resOnlyThic'  --map-width 64 --render --model FullyConv --poet --random-t

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

