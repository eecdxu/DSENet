if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

if [ ! -d "./logs/LongForecasting/univariate" ]; then
    mkdir ./logs/LongForecasting/univariate
fi

seq_len=1920
model_name=CNN_LSTM
root_path_name=/media/wsco/linux_gutai2/XuCd/Dataset/mesa/HDF5/Flow_Thor_Abdo_SpO2/240
data_path_name1=Flow_train_seg.h5
data_path_name2=SpO2_train_seg.h5
data_path_name3=Flow_val_seg.h5
data_path_name4=SpO2_val_seg.h5
data_path_name5=Flow_test.h5
data_path_name6=SpO2_test.h5
data_path_name7=Sleep_event_train_seg.h5
data_path_name8=Sleep_event_val_seg.h5
data_path_name9=Sleep_event_test.h5
data_name=SpO2
random_seed=2024
pred_len=240

# ETTh1, univariate results, pred_len= 24 48 96 192 336 720
# python -u -m debugpy --listen 5678 --wait-for-client ./PatchTST_supervised/run_longExpv2.py \
python -u ./PatchTST_supervised/run_longExpv2.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path1 $data_path_name1 \
  --data_path2 $data_path_name2 \
  --data_path3 $data_path_name3 \
  --data_path4 $data_path_name4 \
  --data_path5 $data_path_name5 \
  --data_path6 $data_path_name6 \
  --data_path7 $data_path_name7 \
  --data_path8 $data_path_name8 \
  --data_path9 $data_path_name9 \
  --model_id CNN_LSTM \
  --model $model_name \
  --data $data_name \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 1 \
  --des 'Flow_SaO2' \
  --itr 1 --batch_size 256 --feature S --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'CNN_LSTM.log