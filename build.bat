@echo off
setlocal EnableDelayedExpansion

:: ============================================================
::  BlackholeSim build script
::  MSVC 2022 / CUDA 13.2 / vcpkg / GTX1650 (CC 7.5)
::
::  Usage:
::    build.bat          -- incremental build
::    build.bat /clean   -- delete build dir, then full rebuild
:: ============================================================

:: -- Resolve project root (strip trailing backslash) ---------
set "PROJECT_DIR=%~dp0"
if "!PROJECT_DIR:~-1!"=="\" set "PROJECT_DIR=!PROJECT_DIR:~0,-1!"
set "BUILD_DIR=!PROJECT_DIR!\build"

set "VCPKG_ROOT=C:\vcpkg"
set "QT6_INST=C:\vcpkg\installed\x64-windows"
set "QT6_TOOLS=C:\vcpkg\installed\x64-windows\tools\Qt6\bin"
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
set "VS_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

:: -- /clean flag -----------------------------------------------
set DO_CLEAN=0
if /i "%~1"=="/clean" set DO_CLEAN=1

:: -- Prerequisite check ----------------------------------------
echo [1/5] Checking prerequisites...
if not exist "!VCPKG_ROOT!\vcpkg.exe" (
    echo [ERROR] vcpkg not found: !VCPKG_ROOT!
    goto :fail
)
if not exist "!CUDA_ROOT!\bin\nvcc.exe" (
    echo [ERROR] CUDA 13.2 not found: !CUDA_ROOT!
    goto :fail
)
if not exist "!VS_VCVARS!" (
    echo [ERROR] Visual Studio 2022 not found
    goto :fail
)
echo   OK

:: -- MSVC environment ------------------------------------------
echo [2/5] Setting up MSVC environment...
call "!VS_VCVARS!" > nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo [ERROR] vcvars64.bat failed
    goto :fail
)
set "PATH=!CUDA_ROOT!\bin;!PATH!"
set "CUDA_PATH=!CUDA_ROOT!"
echo   OK

:: -- Install vcpkg packages (manifest mode, project dir) -------
echo [3/5] Installing vcpkg packages (glm)...
"!VCPKG_ROOT!\vcpkg.exe" install ^
    "--x-manifest-root=!PROJECT_DIR!" ^
    "--x-install-root=!PROJECT_DIR!\vcpkg_installed" ^
    --triplet=x64-windows
if !ERRORLEVEL! neq 0 (
    echo [ERROR] vcpkg install failed
    goto :fail
)
echo   OK

:: -- Clean build dir if /clean was passed ----------------------
if !DO_CLEAN!==1 (
    if exist "!BUILD_DIR!" (
        echo   /clean: removing build directory...
        rmdir /s /q "!BUILD_DIR!"
    )
)

:: -- CMake configure (skip if already configured) --------------
echo [4/5] CMake configure...
set "PKG_INST=!PROJECT_DIR!\vcpkg_installed\x64-windows"
set "PREFIX=!PKG_INST:\=/!;!QT6_INST:\=/!"

if not exist "!BUILD_DIR!\CMakeCache.txt" (
    cmake -B "!BUILD_DIR!" -S "!PROJECT_DIR!" ^
        -G "Visual Studio 17 2022" -A x64 ^
        "-DCMAKE_PREFIX_PATH=!PREFIX!" ^
        "-DCUDA_TOOLKIT_ROOT_DIR=!CUDA_ROOT!" ^
        -DCMAKE_BUILD_TYPE=Release
    if !ERRORLEVEL! neq 0 (
        echo [ERROR] CMake configure failed
        goto :fail
    )
) else (
    echo   Already configured (use /clean to reconfigure)
)
echo   OK

:: -- Build -----------------------------------------------------
echo [5/5] Building Release...
cmake --build "!BUILD_DIR!" --config Release --parallel

if !ERRORLEVEL! neq 0 (
    echo [ERROR] Build failed
    goto :fail
)

:: -- Deploy Qt DLLs and plugins (windeployqt) -----------------
set "EXE=!BUILD_DIR!\Release\BlackholeSim.exe"
set "WINDEPLOYQT=!QT6_TOOLS!\windeployqt.exe"
if exist "!WINDEPLOYQT!" (
    echo Deploying Qt runtime...
    set "PATH=!QT6_INST!\bin;!PATH!"
    "!WINDEPLOYQT!" --release "!EXE!" > nul 2>&1
    echo   Qt runtime deployed
) else (
    echo   [WARN] windeployqt not found, Qt plugins may be missing
)

:: -- Success ---------------------------------------------------
echo.
echo ============================================================
echo  Build succeeded!
echo  Executable: !EXE!
echo ============================================================
echo.
endlocal
exit /b 0

:fail
echo.
echo ============================================================
echo  Build FAILED. Check the error messages above.
echo ============================================================
echo.
endlocal
exit /b 1
