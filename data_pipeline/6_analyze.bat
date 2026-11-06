@echo off
rem Analysis: turn lengths, chunk-size candidates with a suggested min/max, over- and
rem under-redaction, multi-word spans and surprises. Writes summary.md (paste this back)
rem and CSV detail files into %ANALYSIS_DIR%. Safe to re-run.
cd /d "%~dp0"
call settings.bat || goto :end

echo Analysing %ORIGINAL_DIR%, %HUMAN_DIR%, %LLM_DIR%
echo.
%PY% analyze_transcripts.py --original "%ORIGINAL_DIR%" --human "%HUMAN_DIR%" --llm "%LLM_DIR%" --out-dir "%ANALYSIS_DIR%" --labels "%EXTRA_LABELS%" --min-tokens %MIN_TOKENS% --max-tokens-cap %MAX_CAP% --fit-pct %FIT_PCT% --clean-words-pct %CLEAN_WORDS_PCT%
if errorlevel 1 (
  echo.
  echo [FAILED] See the message above.
) else (
  echo.
  echo [DONE] Paste %ANALYSIS_DIR%\summary.md back into the chat.
)

:end
echo.
pause
