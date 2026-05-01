.PHONY: install lint test train

install:
	pip install -r requirements.txt

lint:
	flake8 src/ --max-line-length=100 --exclude=__pycache__

test:
	pytest src/test_pipeline.py -v

train:
	python src/train.py
