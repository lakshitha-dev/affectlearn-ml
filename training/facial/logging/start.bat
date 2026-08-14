@echo off
REM Quick launcher for AffectLearn camera logging menu
REM Double-click this file to start

cd /d "%~dp0"
powershell -NoExit -ExecutionPolicy Bypass -File menu.ps1
pause
