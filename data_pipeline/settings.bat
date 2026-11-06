@echo off
rem ======================================================================
rem settings.bat -- shared settings for all the .bat files in this folder
rem Edit the values below and save. You do not run this file yourself.
rem
rem The transcript folders are built as  data\DATA_SET\<folder name>.
rem Change DATA_SET to switch between data sets (e.g. synthetic or real), and
rem change a folder NAME if it differs on the machine. Spaces are fine.
rem Do NOT end a path with a backslash. For a fully custom path, overwrite the
rem *_DIR line, e.g.  set "ORIGINAL_DIR=D:\research\transcripts\original"
rem ======================================================================

rem --- Data set: the subfolder of data\ that holds the three transcript folders ---
set "DATA_SET=synthetic"

rem --- Folder names inside data\%DATA_SET%\ (each holds .txt files with the SAME names) ---
set "ORIGINAL_NAME=original"
set "HUMAN_NAME=human_redacted"
set "LLM_NAME=llm_redacted"

rem --- Transcript folders (built from the settings above; do not edit) ---
set "ORIGINAL_DIR=..\data\%DATA_SET%\%ORIGINAL_NAME%"
set "HUMAN_DIR=..\data\%DATA_SET%\%HUMAN_NAME%"
set "LLM_DIR=..\data\%DATA_SET%\%LLM_NAME%"

rem --- Extra speaker role names, comma separated, e.g. Student,Guest ---
rem Coach, Teacher and Speaker X are always recognised. Leave empty if none.
set "EXTRA_LABELS="

rem --- Results folder for this data set: catalog, samples and analysis go here ---
rem Built from DATA_SET, so synthetic and real runs never overwrite each other.
set "RESULTS_DIR=..\results\%DATA_SET%"

rem --- Analysis output folder (5_census.bat and 6_analyze.bat write here) ---
set "ANALYSIS_DIR=%RESULTS_DIR%\analysis"

rem --- Step 1: chunk size, in words ---
rem Shorter turns with no redactions are left out; longer turns are split.
set "MIN_TOKENS=50"
set "MAX_TOKENS=600"

rem --- Analysis suggestion thresholds (6_analyze.bat only) ---
rem Largest --max-tokens the suggestion may pick (the LM Studio limit).
set "MAX_CAP=600"
rem Percent of kept turns that must fit unsplit.
set "FIT_PCT=90"
rem Percent of redaction-free words the --min-tokens cut may drop.
set "CLEAN_WORDS_PCT=10"

rem --- Step 2: which sample to make, and step 3/4: which sample to generate from ---
rem Use a NEW name for every new sample (existing samples are never overwritten).
rem N_CHUNKS cannot be more than the number of chunks step 1 reported.
set "SAMPLE_NAME=sample_001"
set "N_CHUNKS=30"
set "SEED=42"

rem LM Studio address, model and retries are set at the top of run_generation.py.

rem ===================== do not edit below this line =====================
set "CATALOG=%RESULTS_DIR%\catalog.csv"
set "SAMPLE_DIR=%RESULTS_DIR%\samples\%SAMPLE_NAME%"

rem Find Python 3.10+: the "py" launcher first, then "python".
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo [ERROR] Python was not found. Install Python 3.10 or newer from python.org
  echo         and tick "Add python.exe to PATH" during setup.
  exit /b 1
)
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
  echo [ERROR] Python 3.10 or newer is required. This one is:
  %PY% --version
  exit /b 1
)
exit /b 0
