mkdir ../results/
CUDA_VISIBLE_DEVICES=0 python train.py \
"model ./models/convtasnet_SSL_FiLM/model.yaml" \
"dataloader ./dataloader/dataloader_train.yaml" \
"hyparam ./hyparam/train.yaml" \
"learner ./hyparam/learner.yaml" \
"logger ./hyparam/logger.yaml"\