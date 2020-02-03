# ESPnet RNNT 모델 Jit script Deploy

## 실행 스크립트 resave_torchjit.py 사용법
### 구조 
python resave_torchjit.py --model-json <json path> --model-torch <model checkpoint path> --output <output .pt file path>
### 예시
python resave_torchjit.py --model-json 20200128/model.json --model-torch 20200128/model.loss.best --output espnet_nvidia_20200128.pt