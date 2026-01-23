@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:menu
echo ========================================
echo    FFmpeg视频分段处理脚本
echo ========================================
echo.

REM 清除之前设置的变量
set "input_folder="
set "crf_value="
set "output_folder="
set "process_mp4_only="

echo 请输入包含视频文件的文件夹路径：
echo （可以直接拖放文件夹到此处）
set "input_folder="
set /p "input_folder=输入文件夹: "

REM 移除路径两端的引号（如果是从拖放得到的）
if defined input_folder (
    set "input_folder=%input_folder:"=%"
)

REM 检查文件夹是否存在
if not defined input_folder (
    echo 错误：未输入路径！
    echo.
    timeout /t 3 >nul
    cls
    goto menu
)

if not exist "%input_folder%\" (
    echo 错误：文件夹不存在！
    echo.
    timeout /t 3 >nul
    cls
    goto menu
)

echo.
echo 请输入CRF值（默认14，范围0-51，数值越小质量越好）：
set "crf_value="
set /p "crf_value=CRF值: "

if not defined crf_value (
    set "crf_value=14"
)

echo.
echo 请输入输出文件夹路径（留空则在原文件夹创建output文件夹）：
set "output_folder="
set /p "output_folder=输出文件夹: "

if not defined output_folder (
    set "output_folder=%input_folder%\output"
) else (
    set "output_folder=%output_folder:"=%"
)

REM 创建输出文件夹
if not exist "%output_folder%" (
    mkdir "%output_folder%"
    echo 已创建输出文件夹：%output_folder%
)

echo.
echo 是否只处理MP4文件？
echo 1. 是（默认）
echo 2. 否（处理所有视频文件）
set "process_mp4_only="
set /p "process_mp4_only=请选择 (1/2): "

if "%process_mp4_only%"=="2" (
    set "file_filter=*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.mpg *.mpeg *.webm *.ts *.m4v *.3gp *.f4v"
) else (
    set "file_filter=*.mp4"
)

echo.
echo ========================================
echo 配置摘要：
echo 输入文件夹： %input_folder%
echo 输出文件夹： %output_folder%
echo CRF值： %crf_value%
echo 处理文件类型： %file_filter%
echo ========================================
echo.

set "confirm="
set /p "confirm=是否开始处理？(Y/N): "
if /i not "%confirm%"=="Y" (
    echo 已取消操作
    echo.
    timeout /t 2 >nul
    cls
    goto menu
)

echo.
echo 开始处理视频文件...
echo.

setlocal enabledelayedexpansion
set "file_count=0"
set "success_count=0"
set "fail_count=0"

REM 获取文件总数用于显示进度
set /a total_files=0
for %%f in ("%input_folder%\%file_filter%") do (
    set /a total_files+=1
)

if %total_files% equ 0 (
    echo 错误：找不到匹配的视频文件！
    echo.
    timeout /t 3 >nul
    cls
    goto menu
)

echo 找到 %total_files% 个视频文件
echo.

REM 处理每个视频文件
for %%f in ("%input_folder%\%file_filter%") do (
    set /a file_count+=1

    REM 获取文件名（不带扩展名）
    set "filename=%%~nf"

    echo [!file_count!/%total_files%] 正在处理: %%~nxf

    REM 执行FFmpeg命令
    REM 注意：输出文件名格式为：原文件名_001.mp4，原文件名_002.mp4等
    ffmpeg -i "%%f" -c:v libx264 -crf %crf_value% -c:a aac -map 0 -force_key_frames "expr:gte(t,n_forced*2)" -segment_time 2 -reset_timestamps 1 -f segment "%output_folder%\!filename!_%%03d.mp4"

    if errorlevel 1 (
        echo     处理失败 ✗
        set /a fail_count+=1
    ) else (
        echo     处理成功 ✓
        set /a success_count+=1
    )

    echo.
)

echo ========================================
echo 处理完成！
echo 总计处理: %file_count% 个文件
echo 成功: %success_count% 个
echo 失败: %fail_count% 个
echo 输出文件夹: %output_folder%
echo ========================================
echo.

set "view_output="
set /p "view_output=是否打开输出文件夹？(Y/N): "
if /i "%view_output%"=="Y" (
    explorer "%output_folder%"
)

echo.
pause
cls
goto menu