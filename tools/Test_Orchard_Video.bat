@echo off
REM ===========================================================
REM   AgriBot - Test an ORCHARD video
REM   (markerless alley-following + obstacle detection)
REM   Drag a video file ONTO this .bat, or double-click and
REM   paste the video path.
REM ===========================================================
cd /d "%~dp0.."

set "video=%~1"
if not defined video set /p video=Drag your orchard video here (or paste its path) then press Enter:

if not defined video goto :novideo

echo.
echo ===========================================================
echo  Testing ORCHARD video:  %video%
echo.
echo  - Blue line  = detected alley centre (steering target)
echo  - Red tint   = tree walls
echo  - RED BOX + "OBSTACLE" = person/vehicle/animal -> STOP
echo  - "ROW END - TURN" appears at the headland
echo  - Dashboard: http://localhost:5000   (Q in window = stop)
echo ===========================================================
echo.

python main.py --video "%video%" --config config/orchard.yaml
goto :done

:novideo
echo No video provided. Drag a video file onto this .bat next time.

:done
echo.
pause
