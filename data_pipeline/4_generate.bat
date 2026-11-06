@echo off
rem Step 3 of 3: generate fake pairs for samples\SAMPLE_NAME using LM Studio.
rem Writes generated.jsonl, failures.jsonl and step_log.txt into that folder.
rem Safe to stop with Ctrl+C (answer Y to "Terminate batch job?");
rem running this file again continues where it stopped.
cd /d "%~dp0"
call settings.bat || goto :end

%PY% run_generation.py --sample-dir "%SAMPLE_DIR%"
if errorlevel 1 (
  echo.
  echo [STOPPED] See the message above. Run this file again to continue.
) else (
  echo.
  echo [DONE] Results are in %SAMPLE_DIR%\generated.jsonl
)

:end
echo.
pause
