PYTHON ?= python3

.PHONY: generate-data train-cbf validate-cbf train-diffusion-cbf smoke

generate-data:
	$(PYTHON) trainingDataGen.py

train-cbf:
	$(PYTHON) trainNN.py

validate-cbf:
	$(PYTHON) validateNN.py

train-diffusion-cbf:
	$(PYTHON) diffusion_cbf.py --epochs 300 --device cpu

smoke:
	$(PYTHON) trainingDataGen.py --T 1 --dt 0.05 --output /tmp/safe_trajectories_test.pt
	$(PYTHON) trainNN.py --data /tmp/safe_trajectories_test.pt --epochs 3 --batch-size 8 --model-out /tmp/cbf_model_test.pth --onnx-out /tmp/cbf_model_test.onnx
	$(PYTHON) validateNN.py --model /tmp/cbf_model_test.pth --T 1 --dt 0.05 --output /tmp/safe_trajectories_learned_test.pt
