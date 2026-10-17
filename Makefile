# Kaizen Labs pipeline. Every step takes CLIENT=<key> (default: demo).
#   make sample && make forecast && make plan && make dashboard
#   make ingest CLIENT=example_15min

CLIENT ?= demo
PY ?= python
PORT ?= 8501

.PHONY: help install sample ingest validate signals forecast plan dashboard report all test screenshot site clean

help:
	@echo "make install                 install the package and test tools"
	@echo "make sample                  generate Demo Café Group sample data in data/sample/"
	@echo "make ingest CLIENT=<key>     raw POS export -> validated contract files (uses clients/<key>/mapping.yaml)"
	@echo "make validate CLIENT=<key>   check a client's input files against the data contract"
	@echo "make signals CLIENT=<key>    fetch and cache public holidays, school holidays, weather"
	@echo "make forecast CLIENT=<key>   signals, then back-test and 7-day hourly forecast"
	@echo "make plan CLIENT=<key>       suggested orders, staffing, gaps, savings estimate"
	@echo "make dashboard CLIENT=<key>  open the dashboard on http://localhost:$(PORT)"
	@echo "make report CLIENT=<key>     one-page PDF summary"
	@echo "make test                    run the tests"

install:
	$(PY) -m pip install -e ".[dev]"

sample:
	$(PY) -m kaizen.sample $(if $(END),--end $(END))

ingest:
	$(PY) -m kaizen.ingest --client $(CLIENT)

validate:
	$(PY) -m kaizen.contract --client $(CLIENT)

# The only step that uses the internet. Everything after it reads cached files.
signals:
	$(PY) -m kaizen.signals --client $(CLIENT) $(if $(OFFLINE),--offline)

forecast: signals
	$(PY) -m kaizen.forecast --client $(CLIENT)

plan:
	$(PY) -m kaizen.plan --client $(CLIENT)

dashboard:
	$(PY) -m streamlit run dashboard/app.py --server.port $(PORT) -- --client $(CLIENT)

report:
	$(PY) -m kaizen.report --client $(CLIENT)

all: sample forecast plan report

test:
	$(PY) -m pytest -q

# Refresh the website's dashboard screenshot (needs `pip install playwright`; dashboard must be running).
screenshot:
	$(PY) scripts/screenshot.py http://localhost:$(PORT) site/assets/dashboard.png 1100 light 1640

# Preview the website locally on http://localhost:8000
site:
	cd site && $(PY) -m http.server 8000

clean:
	rm -rf data/sample data/outputs data/clients/*/inputs data/clients/*/outputs
