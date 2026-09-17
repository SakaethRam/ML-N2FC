# NCFN — CPU image
# Runs the 10-block pipeline via run_pipeline.py (see docs/SETUP_AND_USAGE.md
# for why the block files can't be imported directly).

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the 10 block files, the runner, and docs.
COPY . /app

CMD ["python", "run_pipeline.py"]
