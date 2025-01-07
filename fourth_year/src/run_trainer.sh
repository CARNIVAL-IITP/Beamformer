


rm -rf ./wandb/
device=0
echo "device: $device"

CUDA_VISIBLE_DEVICES=$device python train.py \
"model ./models/EABNET/model.yaml" \
"dataloader ./dataloader/dataloader_train.yaml" \
"hyparam ./hyparam/train.yaml" \
"learner ./hyparam/learner.yaml" \
"logger ./hyparam/logger.yaml"\

rm -rf ./wandb/

bash run_tester.sh