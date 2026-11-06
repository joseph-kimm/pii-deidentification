@echo off
rem Check before generating: LM Studio answers, the model returns JSON,
rem and samples\SAMPLE_NAME\chunks.jsonl can be read. Generates nothing.
cd /d "%~dp0"
call settings.bat || goto :end

%PY% run_generation.py --check --sample-dir "%SAMPLE_DIR%"
if errorlevel 1 (
  echo.
  echo [FAILED] See the message above. Is the LM Studio server started with a model loaded?
) else (
  echo.
  echo [DONE] All checks passed. Next: 4_generate.bat
)

:end
echo.
pause
