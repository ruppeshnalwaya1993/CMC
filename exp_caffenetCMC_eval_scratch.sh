EXP="exp_caffenetCMC_eval_scratch"
mkdir -p ${EXP}
mkdir -p ${EXP}/tb_path
MODEL="exp_caffenetCMC/memory_nce_caffenet_lr_0.001_decay_0.0001_bsz_128/ckpt_epoch_300.pth"
EPOCHS=300

echo ${0}
echo ${1}

rm ${EXP}/${1}_${SPLIT}.txt

for i in {1..3}; do
SPLIT=$i

TCLLIBPATH=/usr/lib/tcltk/x86_64-linux-gnu CUDA_VISIBLE_DEVICES=4,5,6,7 unbuffer python LinearProbing.py --data_folder /dev/shm/UCF101/UCF101_split${SPLIT}/ --model_path ${MODEL}  --tb_path ${EXP}/tb_path/   --save_path ${EXP}  --model caffenet --epochs  ${EPOCHS} --scratch 2>&1 | tee -a ${EXP}/${1}_${SPLIT}.txt 

TCLLIBPATH=/usr/lib/tcltk/x86_64-linux-gnu CUDA_VISIBLE_DEVICES=4,5,6,7 unbuffer python LinearProbing.py --data_folder /dev/shm/UCF101/UCF101_split${SPLIT}/ --model_path ${MODEL}  --tb_path ${EXP}/tb_path/   --save_path ${EXP}  --model caffenet  --resume ${EXP}/calibrated_memory_nce_caffenet_lr_0.001_decay_0.0001_bsz_128_bsz_128_lr_0.01_decay_0.0005/ckpt_epoch_${EPOCHS}.pth --ten_crop --evaluate 2>&1 | tee -a ${EXP}/${1}_${SPLIT}.txt

done

