@echo off
rem ======================================================================
rem settings.bat -- shared settings for 1_build_catalog.bat to 4_generate.bat
rem Edit the values below and save. You do not run this file yourself.
rem
rem Paths are relative to this folder. Absolute paths also work, e.g.
rem   set "ORIGINAL_DIR=D:\research\transcripts\original"
rem Spaces in paths are fine. Do NOT end a path with a backslash.
rem ======================================================================

rem --- Transcript folders: .txt files with the SAME names in all three ---
set "ORIGINAL_DIR=..\data\synthetic\original"
set "HUMAN_DIR=..\data\synthetic\human_redacted"
set "LLM_DIR=..\data\synthetic\llm_redacted"

rem --- Step 1: chunk size, in words ---
rem Shorter turns with no redactions are left out; longer turns are split.
set "MIN_TOKENS=50"
set "MAX_TOKENS=600"

rem --- Step 2: which sample to make, and step 3/4: which sample to generate from ---
rem Use a NEW name for every new sample (existing samples are never overwritten).
rem N_CHUNKS cannot be more than the number of chunks step 1 reported.
set "SAMPLE_NAME=sample_001"
set "N_CHUNKS=30"
set "SEED=42"

rem LM Studio address, model and retries are set at the top of run_generation.py.

rem ===================== do not edit below this line =====================
set "CATALOG=..\results\real_data\catalog.csv"
set "SAMPLE_DIR=..\results\real_data\samples\%SAMPLE_NAME%"

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
