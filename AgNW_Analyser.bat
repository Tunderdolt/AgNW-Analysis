@echo off
REM AgNW_Analyser.bat
REM Double-click this file to run the AgNW batch analysis.
REM Place this file in the same folder as the three Python scripts.
REM
REM The first time you run this it will ask you to select a folder.
REM Results will be saved as a zip file in the folder you choose.

title AgNW Batch Analyser

REM ── Find Python ──────────────────────────────────────────────────────────────
REM Try common locations in order. Edit the PYTHON line below if yours is elsewhere.

SET PYTHON=
IF EXIST "%USERPROFILE%\.julia\conda\3\x86_64\python.exe" (
    SET PYTHON=%USERPROFILE%\.julia\conda\3\x86_64\python.exe
)
IF NOT DEFINED PYTHON (
    FOR /F "delims=" %%i IN ('where python 2^>nul') DO (
        SET PYTHON=%%i
        GOTO :found_python
    )
)
:found_python

IF NOT DEFINED PYTHON (
    echo.
    echo ERROR: Python not found.
    echo Please install Python from python.org and tick "Add to PATH".
    echo.
    pause
    exit /b 1
)

REM ── Ask user for database folder ──────────────────────────────────────────────
echo.
echo ================================================================
echo   AgNW Batch Analyser
echo ================================================================
echo.
echo   Python : %PYTHON%
echo   Scripts: %~dp0
echo.
echo   Drag and drop your database folder into this window,
echo   then press Enter.
echo   (Or type the full path manually)
echo.
SET /P DB_FOLDER="Database folder: "

REM Strip surrounding quotes if drag-dropped
SET DB_FOLDER=%DB_FOLDER:"=%

IF NOT EXIST "%DB_FOLDER%" (
    echo.
    echo ERROR: Folder not found: %DB_FOLDER%
    echo.
    pause
    exit /b 1
)

REM ── Ask for output folder ─────────────────────────────────────────────────────
echo.
echo   Where should the results zip be saved?
echo   Press Enter to save alongside the database folder.
echo.
SET /P OUT_FOLDER="Output folder (or Enter for default): "
SET OUT_FOLDER=%OUT_FOLDER:"=%

REM ── Run ───────────────────────────────────────────────────────────────────────
echo.
echo   Starting analysis...
echo   This may take several minutes per image. Do not close this window.
echo.

SET SCRIPT=%~dp0run_batch_analysis.py
SET "PYTHONIOENCODING=utf-8"

IF "%OUT_FOLDER%"=="" (
    "%PYTHON%" "%SCRIPT%" "%DB_FOLDER%"
) ELSE (
    "%PYTHON%" "%SCRIPT%" "%DB_FOLDER%" --output_dir "%OUT_FOLDER%"
)

echo.
IF %ERRORLEVEL% EQU 0 (
    echo   Done! Check the output folder for your results zip.
) ELSE (
    echo   Something went wrong. Check the messages above for details.
)
echo.
pause
