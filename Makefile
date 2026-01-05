init:
	poetry run python -m originmap.cli init

download:
	poetry run python -m originmap.cli download

ingest:
	poetry run python -m originmap.cli ingest

metrics:
	poetry run python -m originmap.cli metrics

visualize:
	poetry run python -m originmap.cli visualize

report:
	poetry run python -m originmap.cli report

observe:
	poetry run python -m originmap.cli observe

hypothesis:
	poetry run python -m originmap.cli hypothesis O-D2 --n 500

hypothesis-stratified:
	poetry run python -m originmap.cli hypothesis O-D3 --n 500

full-cycle:
	poetry run python -m originmap.cli full-cycle

# Run full pipeline in sequence
pipeline:
	make download && make ingest && make metrics && make visualize && make report && make observe

# Full scientific analysis
science:
	make pipeline && make hypothesis

test:
	poetry run pytest -v
