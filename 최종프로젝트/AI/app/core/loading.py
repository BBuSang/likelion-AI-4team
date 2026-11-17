"""무거운 모델(임베딩·탐지)을 처음 읽을 때 쓰는 잠금 하나.

서버 시작 때 임베딩 모델을 백그라운드에서 읽는 동안 첫 질문이 들어오면, 질문 쪽이 탐지 모델을 동시에 읽으면서
두 스레드가 transformers 를 같이 import 해 ImportError 가 났음 (첫 질문 502, 2026-11-17).
모델 읽기를 이 잠금 하나로 묶어 한 번에 하나씩만 읽게 함. 다시 들어와도 되게 RLock.
"""
import threading

MODEL_LOAD_LOCK = threading.RLock()
