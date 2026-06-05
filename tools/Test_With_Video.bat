@echo off
REM ===========================================================
REM   AgriBot - Test with a recorded video
REM   Drag a video file (mp4/avi/mov) ONTO this .bat to run it,
REM   or just double-click and type/paste the video path.
REM ===========================================================
cd /d "%~dp0.."

set "video=%~1"
if not defined video set /p video=Drag your video onto this window (or paste its path) then press Enter:

if not defined video goto :novideo

echo.
echo ===========================================================
echo  Testing with video:  %video%
echo.
echo  - A window will open showing the video with detections.
echo  - Open  http://localhost:5000  in a browser for the dashboard.
echo  - Press Q in the video window to stop.
echo ===========================================================
echo.

python main.py --video "%video%"
goto :done

:novideo
echo.
echo No video provided. Drag a video file onto this .bat next time.

:done
echo.
pause
