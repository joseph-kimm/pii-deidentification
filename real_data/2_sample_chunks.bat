@echo off
rem Step 2 of 3: randomly pick N_CHUNKS chunks from catalog.csv into samples\SAMPLE_NAME.
rem Fails on purpose if that sample folder already exists: change SAMPLE_NAME in settings.bat.
cd /d "%~dp0"
call settings.bat || goto :end

echo Sampling %N_CHUNKS% chunks, seed %SEED%, into %SAMPLE_DIR%
echo.
%PY% sample_chunks.py --catalog "%CATALOG%" --original "%ORIGINAL_DIR%" --human "%HUMAN_DIR%" --llm "%LLM_DIR%" --out-dir "%SAMPLE_DIR%" --n %N_CHUNKS% --seed %SEED%
if errorlevel 1 (
  echo.
  echo [FAILED] See the message above.
) else (
  echo.
  echo [DONE] Next: start the LM Studio server, then 3_check_lmstudio.bat
)

:end
echo.
pause
