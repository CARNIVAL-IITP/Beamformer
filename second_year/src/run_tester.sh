rm -rf ./wandb/

CUDA_VISIBLE_DEVICES=0 python inference.py \
"model ./models/convtasnet_SSL_FiLM/model.yaml" \
"dataloader ./dataloader/dataloader_test.yaml" \
"hyparam ./hyparam/test.yaml" \
"learner ./hyparam/learner.yaml" \
"logger ./hyparam/logger.yaml"\