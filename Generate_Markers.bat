@echo off
setlocal
cd /d "%~dp0"

echo ==================================================
echo     AgriBot  -  ArUco Marker Generator
echo ==================================================
echo.
echo This creates printable marker images in the
echo "markers" folder. Print the ones you need.
echo.
echo   - Rows 1..N        = normal row-end markers
echo   - LAST-ROW (249)   = far end of the LAST row
echo   - STOP     (248)   = near end of the LAST row
echo.

set "rows="
set /p rows=How many crop rows does your field have (press Enter for 8)?
if "%rows%"=="" set "rows=8"

echo.
echo Clearing any old markers so only this field's markers remain ...
if exist "markers\aruco_marker_*.png" del /q "markers\aruco_marker_*.png"

echo Generating row markers 1..%rows% plus LAST-ROW and STOP ...
echo.

python tools\generate_marker.py --range 1 %rows% --out markers
python tools\generate_marker.py --ids 249,248 --out markers

echo.
echo --------------------------------------------------
echo  Done!  Opening the "markers" folder...
echo  Print each at least 20 cm x 20 cm and laminate.
echo --------------------------------------------------
start "" "%~dp0markers"

echo.
pause
endlocal
