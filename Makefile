PY = python3

.PHONY: all install run mock test

all: install run

install:
	pip install -r requirements.txt

# 真实云 API 跑基线(需先 cp .env.example .env 并填 key)
run:
	$(PY) -m src.main --config configs/models.yaml --probes probes/ --out results/

# 离线跑通(无需 API key,模拟"对 direct-injection/system-leak 脆弱"的假模型)
mock:
	$(PY) -m src.main --mock --config configs/models.yaml --probes probes/ --out results/

# 带 guard 的 2×2 复测 + 良性 FPR
test:
	$(PY) -m src.main --mock --guard --benign benign/benign_inputs.jsonl \
		--config configs/models.yaml --probes probes/ --out results/
