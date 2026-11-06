@echo off
rem Step 1 of 3: read the transcript folders set in settings.bat and write catalog.csv.
rem Re-running overwrites catalog.csv.
cd /d "%~dp0"
call settings.bat || goto :end

echo Building %CATALOG% from:
echo   original: %ORIGINAL_DIR%
echo   human:    %HUMAN_DIR%
echo   llm:      %LLM_DIR%
echo.
%PY% build_catalog.py --original "%ORIGINAL_DIR%" --human "%HUMAN_DIR%" --llm "%LLM_DIR%" --out "%CATALOG%" --min-tokens %MIN_TOKENS% --max-tokens %MAX_TOKENS% --tokenizer words --labels "%EXTRA_LABELS%"
if errorlevel 1 (
  echo.
  echo [FAILED] See the message above.
) else (
  echo.
  echo [DONE] Next: 2_sample_chunks.bat
)

:end
echo.
pause
