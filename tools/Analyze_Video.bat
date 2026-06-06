@echo off
REM ===========================================================
REM   AgriBot - Analyze a video (headless quality check)
REM   Runs the FULL pipeline over the video, writes an annotated
REM   output video with the STATE banner, and prints every
REM   DRIVING / TURNING / STOPPED transition.
REM   Drag a video file ONTO this .bat, or paste its path.
REM ===========================================================
cd /d "%~dp0.."

set "video=%~1"
if not defined video set /p video=Drag your video here (or paste its path) then press Enter:

if not defined video goto :novideo

echo.
echo Analyzing %video% with config\orchard.yaml ...
echo (Writes <video>_analyzed.mp4 and prints state transitions.)
echo.
python tools\analyze_video.py "%video%" --config config\orchard.yaml
goto :done

:novideo
echo No video provided. Drag a video file onto this .bat next time.

:done
echo.
pause
