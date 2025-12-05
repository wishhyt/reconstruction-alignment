# accelerate launch --num_processes 1 \
#   scripts/evaluation/compbench_infer.py \
#   configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py \
#   --checkpoint /home/jixie/LEGACY/OpenUni/checkpoints/openuni_l_internvl3_2b_sana_1_6b_512_hf_text2image23m.pth \
#   --batch_size 1 \
#   --data /home/jixie/SRUM/val_comp.json \
#   --output /home/jixie/comp_eval/2_1.6_ori \
#   --cfg_scale 4.5 \
#   --num_steps 20 \
#   --height 512 --width 512 \
#   --seed 42

# accelerate launch --num_processes 1 \
#   scripts/evaluation/compbench_infer.py \
#   configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py \
#   --checkpoint /home/jixie/LEGACY/OpenUni/old_results/OpenUni_3.6b_ReAlign.pth \
#   --batch_size 1 \
#   --data /home/jixie/SRUM/val_comp.json \
#   --output /home/jixie/comp_eval/2_1.6_reca \
#   --cfg_scale 4.5 \
#   --num_steps 20 \
#   --height 512 --width 512 \
#   --seed 42

# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp2_1_0.6/iter_2000.pth  --batch_size 2  --output exp2_1_0.6_2000 --height 512 --width 512 --seed 42 
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp2_1_0.6/iter_1000.pth  --batch_size 2  --output exp2_1_0.6_1000 --height 512 --width 512 --seed 42 
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp2_1_0.6/iter_3000.pth  --batch_size 2  --output exp2_1_0.6_3000 --height 512 --width 512 --seed 42 
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp2_1_0.6/iter_4000.pth  --batch_size 2  --output exp2_1_0.6_4000 --height 512 --width 512 --seed 42 
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp2_1_0.6/iter_5000.pth  --batch_size 2  --output exp2_1_0.6_5000 --height 512 --width 512 --seed 42 
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp2_1_0.6/iter_6000.pth  --batch_size 2  --output exp2_1_0.6_6000 --height 512 --width 512 --seed 42 
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp2_1_0.6/iter_7000.pth  --batch_size 2  --output exp2_1_0.6_7000 --height 512 --width 512 --seed 42 
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp2_1_0.6/iter_8000.pth  --batch_size 2  --output exp2_1_0.6_8000 --height 512 --width 512 --seed 42 
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp2_1_0.6/iter_7000.pth  --batch_size 2  --output exp2_1_0.6_7000 --height 512 --width 512 --seed 42 
# accelerate launch scripts/evaluation/dpg_bench.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp2_1_0.6/iter_3000.pth  --batch_size 2  --output dpg_exp2_1_0.6_3000 --height 512 --width 512 --seed 42 
# accelerate launch scripts/evaluation/dpg_bench.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp2_1_0.6/iter_4000.pth  --batch_size 2  --output dpg_exp2_1_0.6_4000 --height 512 --width 512 --seed 42 
# accelerate launch scripts/evaluation/dpg_bench.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp2_1_0.6/iter_5000.pth  --batch_size 2  --output dpg_exp2_1_0.6_5000 --height 512 --width 512 --seed 42 
# accelerate launch scripts/evaluation/wise.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp2_1_0.6/iter_5000.pth  --batch_size 2  --output wise_exp2_1_0.6_5000 --height 512 --width 512 --seed 42 --data ../ReAlign/WISE/data/cultural_common_sense.json
# accelerate launch scripts/evaluation/wise.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp2_1_0.6/iter_5000.pth  --batch_size 2  --output wise_exp2_1_0.6_5000 --height 512 --width 512 --seed 42 --data ../ReAlign/WISE/data/natural_science.json 
# accelerate launch scripts/evaluation/wise.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp2_1_0.6/iter_5000.pth  --batch_size 2  --output wise_exp2_1_0.6_5000 --height 512 --width 512 --seed 42 --data ../ReAlign/WISE/data/spatio-temporal_reasoning.json
# accelerate launch scripts/evaluation/wise.py configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint /home/jixie/OpenUni/checkpoints/openuni_l_internvl3_2b_sana_1_6b_512_hf_text2image23m.pth  --batch_size 2  --output wise_2_1.6 --height 512 --width 512 --seed 42 --data ../ReAlign/WISE/data/natural_science.json 
# accelerate launch scripts/evaluation/wise.py configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint /home/jixie/OpenUni/checkpoints/openuni_l_internvl3_2b_sana_1_6b_512_hf_text2image23m.pth  --batch_size 2  --output wise_2_1.6 --height 512 --width 512 --seed 42 --data ../ReAlign/WISE/data/spatio-temporal_reasoning.json

# accelerate launch scripts/evaluation/wise.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint /home/jixie/OpenUni/checkpoints/openuni_b_internvl3_1b_sana_0_6b_512_hf_text2image23m.pth  --batch_size 2  --output wise_1_0.6 --height 512 --width 512 --seed 42 --data ../ReAlign/WISE/data/cultural_common_sense.json
# accelerate launch scripts/evaluation/wise.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint /home/jixie/OpenUni/checkpoints/openuni_b_internvl3_1b_sana_0_6b_512_hf_text2image23m.pth  --batch_size 2  --output wise_1_0.6 --height 512 --width 512 --seed 42 --data ../ReAlign/WISE/data/natural_science.json 
# accelerate launch scripts/evaluation/wise.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint /home/jixie/OpenUni/checkpoints/openuni_b_internvl3_1b_sana_0_6b_512_hf_text2image23m.pth  --batch_size 2  --output wise_1_0.6 --height 512 --width 512 --seed 42 --data ../ReAlign/WISE/data/spatio-temporal_reasoning.json



# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp3_1_0.6/iter_2000.pth  --batch_size 2  --output exp3_1_0.6_2000 --height 512 --width 512 --seed 42 --base checkpoints/openuni_b_internvl3_1b_sana_0_6b_512_hf_blip3o60k.pth
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp3_1_0.6/iter_1000.pth  --batch_size 2  --output exp3_1_0.6_1000 --height 512 --width 512 --seed 42 --base checkpoints/openuni_b_internvl3_1b_sana_0_6b_512_hf_blip3o60k.pth
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp3_1_0.6/iter_3000.pth  --batch_size 2  --output exp3_1_0.6_3000 --height 512 --width 512 --seed 42 --base checkpoints/openuni_b_internvl3_1b_sana_0_6b_512_hf_blip3o60k.pth
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp3_1_0.6/iter_4000.pth  --batch_size 2  --output exp3_1_0.6_4000 --height 512 --width 512 --seed 42 --base checkpoints/openuni_b_internvl3_1b_sana_0_6b_512_hf_blip3o60k.pth
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_b_internvl3_1b_sana_0_6b_512_hf.py --checkpoint work_dirs/exp3_1_0.6/iter_5000.pth  --batch_size 2  --output exp3_1_0.6_5000 --height 512 --width 512 --seed 42 --base checkpoints/openuni_b_internvl3_1b_sana_0_6b_512_hf_blip3o60k.pth
# accelerate launch  --num_processes 5 scripts/evaluation/gen_eval.py configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint /home/jixie/LEGACY/OpenUni/old_results/OpenUni_3.6b_ReAlign.pth  --batch_size 1  --output 2_1.6_reca --height 512 --width 512 --seed 42
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp1_2_1.6/iter_2000.pth  --batch_size 1  --output exp1_2_1.6_2000 --height 512 --width 512 --seed 42
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp1_2_1.6/iter_3000.pth  --batch_size 1  --output exp1_2_1.6_3000 --height 512 --width 512 --seed 42
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp1_2_1.6/iter_6000.pth  --batch_size 1  --output exp1_2_1.6_6000 --height 512 --width 512 --seed 42
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp1_2_1.6/iter_7000.pth  --batch_size 1  --output exp1_2_1.6_7000 --height 512 --width 512 --seed 42

# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint /home/jixie/OpenUni/checkpoints/openuni_l_internvl3_2b_sana_1_6b_512_hf_text2image23m.pth  --batch_size 1  --output 2_1.6_ori --height 512 --width 512 --seed 42
# accelerate launch scripts/evaluation/dpg_bench.py  configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint /home/jixie/OpenUni/checkpoints/openuni_l_internvl3_2b_sana_1_6b_512_hf_text2image23m.pth  --batch_size 1 --output dpg_2_1.6_ori --height 512 --width 512 --seed 42

# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp3_2_1.6/iter_1000.pth  --batch_size 1  --output exp3_2_1.6_1000 --height 512 --width 512 --seed 42 --base /home/jixie/OpenUni/checkpoints/openuni_l_internvl3_2b_sana_1_6b_512_hf_blip3o60k.pth
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp3_2_1.6/iter_2000.pth  --batch_size 1  --output exp3_2_1.6_2000 --height 512 --width 512 --seed 42 --base /home/jixie/OpenUni/checkpoints/openuni_l_internvl3_2b_sana_1_6b_512_hf_blip3o60k.pth
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint /home/jixie/OpenUni/checkpoints/openuni_l_internvl3_2b_sana_1_6b_512_hf_blip3o60k.pth  --batch_size 1  --output 2_1.6_ori_3o --height 512 --width 512 --seed 42 
# accelerate launch scripts/evaluation/dpg_bench.py  configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint /home/jixie/OpenUni/checkpoints/openuni_l_internvl3_2b_sana_1_6b_512_hf_text2image23m.pth  --batch_size 1 --output dpg_2_1.6_ori --height 512 --width 512 --seed 42



# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp2_2_1.6/iter_3000.pth  --batch_size 1  --output exp2_2_1.6_3000 --height 512 --width 512 --seed 42
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp2_2_1.6/iter_6000.pth  --batch_size 1  --output exp2_2_1.6_6000 --height 512 --width 512 --seed 42
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp2_2_1.6/iter_7000.pth  --batch_size 1  --output exp2_2_1.6_7000 --height 512 --width 512 --seed 42
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp2_2_1.6/iter_3000.pth  --batch_size 1  --output exp2_2_1.6_3000 --height 512 --width 512 --seed 42
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp2_2_1.6/iter_4000.pth  --batch_size 1  --output exp2_2_1.6_4000 --height 512 --width 512 --seed 41
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp2_2_1.6/iter_5000.pth  --batch_size 1  --output exp2_2_1.6_5000 --height 512 --width 512 --seed 42

# accelerate launch scripts/evaluation/dpg_bench.py  configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp2_2_1.6/iter_4000.pth  --batch_size 1 --output dpg_exp2_2_1.6_4000 --height 512 --width 512 --seed 42
# accelerate launch scripts/evaluation/wise.py  configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp2_2_1.6/iter_4000.pth  --batch_size 12 --output wise_exp2_2_1.6_4000 --height 512 --width 512 --seed 42 --data ../WISE/data/cultural_common_sense.json
# accelerate launch scripts/evaluation/wise.py  configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp2_2_1.6/iter_4000.pth  --batch_size 12 --output wise_exp2_2_1.6_4000 --height 512 --width 512 --seed 42 --data ../WISE/data/natural_science.json 
# accelerate launch scripts/evaluation/wise.py  configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp2_2_1.6/iter_4000.pth  --batch_size 12 --output wise_exp2_2_1.6_4000 --height 512 --width 512 --seed 42 --data ../WISE/data/spatio-temporal_reasoning.json

# accelerate launch scripts/evaluation/wise.py  configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp2_2_1.6/iter_3000.pth  --batch_size 12 --output wise_exp2_2_1.6_3000 --height 512 --width 512 --seed 42 --data ../WISE/data/natural_science.json 
# accelerate launch scripts/evaluation/wise.py  configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp2_2_1.6/iter_3000.pth  --batch_size 12 --output wise_exp2_2_1.6_3000 --height 512 --width 512 --seed 42 --data ../WISE/data/cultural_common_sense.json
# accelerate launch scripts/evaluation/wise.py  configs/models/openuni_l_internvl3_2b_sana_1_6b_512_hf.py --checkpoint work_dirs/exp2_2_1.6/iter_3000.pth  --batch_size 12 --output wise_exp2_2_1.6_3000 --height 512 --width 512 --seed 42 --data ../WISE/data/spatio-temporal_reasoning.json
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_l_internvl3_2b_sana_1_6b_1024_hf.py --checkpoint work_dirs/non_exp1/iter_1000.pth  --batch_size 1  --output non_exp1_1000 --height 1024 --width 1024 --seed 42
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_l_internvl3_2b_sana_1_6b_1024_hf.py --checkpoint work_dirs/non_exp1/iter_2000.pth  --batch_size 1  --output non_exp1_2000 --height 1024 --width 1024 --seed 42
# accelerate launch scripts/evaluation/gen_eval.py configs/models/openuni_l_internvl3_2b_sana_1_6b_1024_hf.py --checkpoint /home/jixie/LEGACY/OpenUni/checkpoints/openuni_l_internvl3_2b_sana_1_6b_1024_hf_blip3o60k.pth  --batch_size 1  --output non_exp0 --height 1024 --width 1024 --seed 42

