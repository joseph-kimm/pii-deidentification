@echo off
rem Census (run first, independent of steps 1-4): check speaker labels and timestamps
rem before trusting the catalog or the analysis. Writes census.txt and census.csv
rem into %ANALYSIS_DIR%\census. Safe to re-run.
cd /d "%~dp0"
call settings.bat || goto :end

echo Running census on %ORIGINAL_DIR%, %HUMAN_DIR%, %LLM_DIR%
echo.
%PY% census.py --original "%ORIGINAL_DIR%" --human "%HUMAN_DIR%" --llm "%LLM_DIR%" --out-dir "%ANALYSIS_DIR%\census" --labels "%EXTRA_LABELS%"
if errorlevel 1 (
  echo.
  echo [FAILED] See the message above.
) else (
  echo.
  echo [DONE] Read %ANALYSIS_DIR%\census\census.txt. Next: 6_analyze.bat
)

:end
echo.
pause
