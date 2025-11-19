PYTHON=python3
VENV_DIR=.venv
PIP=$(VENV_DIR)/bin/pip
PYTHON_VENV=$(VENV_DIR)/bin/python

all: help

help:
	@echo "Comandos disponibles:"
	@echo "  make venv            -> Crea entorno virtual"
	@echo "  make install         -> Instala dependencias"
	@echo "  make lint            -> Lint con pylint"
	@echo "  make test            -> Ejecuta tests"
	@echo "  make clasificar      -> Ejecuta sistema de clasificación (YOLO -> CSV)"
	@echo "  make etl             -> Ejecuta sistema Batch/ETL (CSV -> HDFS/Hive)"
	@echo "  make hive-tables     -> Crea tablas en Hive"

venv:
	@test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

lint:
	$(VENV_DIR)/bin/pylint src

test:
	$(PYTHON_VENV) -m pytest tests

clasificar:
	$(PYTHON_VENV) src/sistema_clasificacion.py

etl:
	$(PYTHON_VENV) src/sistema_batch_etl.py

hive-tables:
	beeline -u "jdbc:hive2://localhost:10000/default" -n yocelyn -f scripts_hive/create_tables.sql
