@echo off
REM ===========================================================
REM   AgriBot - Synthetic SIMULATION (no camera / no video)
REM   Renders a drawn orchard alley and shows the machine:
REM     - follow the grassy alley (markerless),
REM     - STOP for an object in the path,
REM     - U-TURN when a red row-end pole approaches.
REM   Dashboard: http://localhost:5000   (press Q to stop)
REM ===========================================================
cd /d "%~dp0.."

echo.
echo  Starting orchard simulation (close the window or press Q to stop)...
echo.

python main.py --simulate --config config/orchard.yaml

echo.
pause
