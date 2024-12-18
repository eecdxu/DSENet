if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/MESA_Flow_nonan_30s_MGA" ]; then
    mkdir ./logs/MESA_Flow_nonan_30s_MGA
fi

seq_len=960
model_name=PatchTST_MGA
root_path_name=/media/wsco/XuCd/Dataset/mesa/HDF5v2/30s_clsv2
data_path_name1=Flow.h5
data_path_name2=Sleep_event.h5
model_id_name=Flow
data_name=Flow
patch_len=48
pred_len=1
random_seed=2024
#  6 8 66 88 520 1314 2024
for idx in 0
do
for fold in 0 1 2 3 4
do
    # python -u -m debugpy --listen 5679 --wait-for-client ./PatchTST_supervised/run_longExp.py \
    python -u ./PatchTST_supervised/run_longExp_ori.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path1 $data_path_name1 \
      --data_path2 $data_path_name2 \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features S \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 1 \
      --e_layers 3 \
      --n_heads 4 \
      --inner_dim 20 \
      --d_model 320 \
      --d_ff 512 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0.2\
      --embedding_dropout 0.1\
      --patch_len $patch_len\
      --stride 24\
      --des 'Flow' \
      --train_epochs 100\
      --StudyNumber $idx\
      --checkpoints './MESA_checkpoints_nonan_30s_MGA/' \
      --result_path 'result_MESA_Flow_nonan_30s_MGA.txt' \
      --c_kernel 21\
      --c_stride 11\
      --fold $fold\
      --lradj 'TST'\
      --itr 1 --batch_size 256 --learning_rate 0.0001 >logs/MESA_Flow_nonan_30s_MGA/$model_name'_fS'$model_id_name'_sl'$seq_len'_pl'$pred_len'_pl'$patch_len'_Flow_SpO2_SN'$idx'_fold'$fold.log 
done
done