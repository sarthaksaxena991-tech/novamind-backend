@echo off
setlocal

REM change to project dir
cd /d C:\Users\shash\OneDrive\Desktop\ai_assistant_hub\tools\voca_remover

REM activate venv
call .\venv\Scripts\activate

REM 1) rebuild flags + enhance + housekeeping
python automation\run_automation.py

REM (optional) run enhancer separately if you want a second pass
REM python automation\auto_improve.py

endlocal