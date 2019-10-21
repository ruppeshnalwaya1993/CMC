EXP="exp_caffenetCMC"
mkdir -p ${EXP}
mkdir -p ${EXP}/tb_path
echo ${0}
echo ${1}

TCLLIBPATH=/usr/lib/tcltk/x86_64-linux-gnu CUDA_VISIBLE_DEVICES=4,5,6,7 unbuffer python train_CMC.py --data_folder /dev/shm/UCF101/UCF101_split1/ --model_path ${EXP}/ --tb_path ${EXP}/tb_path/ --model caffenet  2>&1 | tee ${EXP}/${1}.txt

