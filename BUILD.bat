@echo OFF 

echo Building INREngine
echo.

choice /C YN /M "Do you want to install Python dependencies (only needed for Py visualization)?"

if errorlevel 2 goto skip_python
if errorlevel 1 goto install_python

:install_python
call install_py_deps.bat
goto setup_vs

:skip_python
echo.
echo Skipping Python dependencies...


echo .
echo Locating Visual Studio Installation...
echo .
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"

if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found! Please ensure Visual Studio 2017 or newer is installed
    pause
    exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VS_PATH=%%i"
)

if not defined VS_PATH (
    echo ERROR: Could not find a Visual Studio installation with C++ tools installed
    pause
    exit /b 1
)

echo Found Visual Studio at: %VS_PATH%
echo Setting up Environment...
echo .

call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"

echo .
echo Starting Build for Visual Studio project...
echo .  

msbuild "3b1b_backprop.sln" /p:Configuration=Training /p:Platform=x64

if %errorlevel% neq 0 (
    echo .
    echo =========================================
    echo BUILD FAILED! Check the errors above
    echo =========================================
    pause
    exit /b %errorlevel%
)

echo . 
echo Build completed successfully
pause