device=0
CUDA_VISIBLE_DEVICES=$device python inference.py \
"model ./models/EABNET/model.yaml" \
"dataloader ./dataloader/dataloader_test.yaml" \
"hyparam ./hyparam/test.yaml" \
"learner ./hyparam/learner.yaml" \
"logger ./hyparam/logger.yaml"\