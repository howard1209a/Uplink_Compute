@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
echo ========================================
echo 高级CMP投影转换工具 - 支持参数配置
echo ========================================

rem ========== 参数配置区 ==========
set DEFAULT_RESOLUTION=1920x960
set DEFAULT_CRF=14
set DEFAULT_INPUT=video_4.mp4
set DEFAULT_OUTPUT_DIR=tiles_output

rem ========== 用户输入区 ==========
echo.
echo 请选择分辨率选项:
echo 1. 1920x960 (标准1K CMP, 瓦片尺寸: 240x240)
echo 2. 2560x1280 (1.3K CMP, 瓦片尺寸: 320x320)
echo 3. 3840x1920 (2K CMP, 瓦片尺寸: 480x480)
echo 4. 自定义分辨率
set /p RES_CHOICE="请选择 (1-4, 默认1): "
if "!RES_CHOICE!"=="" set RES_CHOICE=1

if "!RES_CHOICE!"=="1" (
    set RESOLUTION=1920x960
    set TILE_SIZE=480
) else if "!RES_CHOICE!"=="2" (
    set RESOLUTION=2560x1280
    set TILE_SIZE=640
) else if "!RES_CHOICE!"=="3" (
    set RESOLUTION=3840x1920
    set TILE_SIZE=960
) else if "!RES_CHOICE!"=="4" (
    set /p CUSTOM_RES="请输入自定义分辨率 (格式: 宽度x高度, 如1920x960): "
    set RESOLUTION=!CUSTOM_RES!
    for /f "tokens=1,2 delims=x" %%a in ("!RESOLUTION!") do (
        set WIDTH=%%a
        set HEIGHT=%%b
    )
    set /a TILE_SIZE=!WIDTH!/8
) else (
    set RESOLUTION=!DEFAULT_RESOLUTION!
    set TILE_SIZE=240
)

echo.
set /p INPUT_FILE="输入视频文件 (默认: !DEFAULT_INPUT!): "
if "!INPUT_FILE!"=="" set INPUT_FILE=!DEFAULT_INPUT!

echo.
set /p CRF_VALUE="CRF质量值 (0-51, 越小质量越高, 默认: !DEFAULT_CRF!): "
if "!CRF_VALUE!"=="" set CRF_VALUE=!DEFAULT_CRF!

echo.
set /p OUTPUT_DIR="输出文件夹 (默认: !DEFAULT_OUTPUT_DIR!): "
if "!OUTPUT_DIR!"=="" set OUTPUT_DIR=!DEFAULT_OUTPUT_DIR!

rem ========== 解析分辨率 ==========
for /f "tokens=1,2 delims=x" %%a in ("!RESOLUTION!") do (
    set WIDTH=%%a
    set HEIGHT=%%b
)

echo.
echo ========================================
echo 配置摘要:
echo 输入文件: !INPUT_FILE!
echo 输出分辨率: !RESOLUTION! (宽度: !WIDTH!, 高度: !HEIGHT!)
echo 瓦片尺寸: !TILE_SIZE!x!TILE_SIZE!
echo CRF质量值: !CRF_VALUE!
echo 输出文件夹: !OUTPUT_DIR!
echo ========================================
echo.
set /p CONFIRM="确认配置? (Y/N, 默认Y): "
if /i "!CONFIRM!"=="N" exit /b

rem ========== 创建输出文件夹 ==========
if not exist "!OUTPUT_DIR!" mkdir "!OUTPUT_DIR!"

rem ========== 步骤1: 生成CMP视频 ==========
echo.
echo 步骤1: 生成CMP投影视频 (!RESOLUTION!, 3x2布局)...
set OUTPUT_CMP=!OUTPUT_DIR!\video_cmp_!WIDTH!x!HEIGHT!.mp4

ffmpeg -i "!INPUT_FILE!" -vf "scale=!WIDTH!:!HEIGHT!:flags=lanczos+accurate_rnd+full_chroma_int+full_chroma_inp,format=yuv420p,v360=equirect:c3x2" -c:v libx264 -preset slower -crf !CRF_VALUE! -profile:v high -level 5.1 -x264-params "aq-mode=3:aq-strength=1.0:psy-rd=1.0:psy-rdoq=1.0" -c:a copy -movflags +faststart -pix_fmt yuv420p "!OUTPUT_CMP!"

if errorlevel 1 (
    echo 错误: CMP视频生成失败!
    pause
    exit /b 1
)

echo CMP视频生成成功: !OUTPUT_CMP!
echo.

rem ========== 步骤2: 切割瓦片 ==========
echo 步骤2: 开始切割24个瓦片...
echo ========================================

rem 瓦片坐标定义 (基于CMP 3x2布局)
rem 注意: CMP布局是3列x2行，共6个面，每个面再分成2x2=4个瓦片
rem 我们使用硬编码的坐标来确保正确性

set TILE_INDEX=0

echo 生成面1 (左上区域)...
ffmpeg -i "!OUTPUT_CMP!" -vf "crop=!TILE_SIZE!:!TILE_SIZE!:0:0" -c:v libx264 -preset slower -crf !CRF_VALUE! -c:a copy "!OUTPUT_DIR!\video_4_tile1.mp4"

echo 生成面2 (中上区域)...
ffmpeg -i "!OUTPUT_CMP!" -vf "crop=!TILE_SIZE!:!TILE_SIZE!:!TILE_SIZE!:0" -c:v libx264 -preset slower -crf !CRF_VALUE! -c:a copy "!OUTPUT_DIR!\video_4_tile2.mp4"

echo 生成面3(右上区域)...
set /a X=!TILE_SIZE!*2
ffmpeg -i "!OUTPUT_CMP!" -vf "crop=!TILE_SIZE!:!TILE_SIZE!:!X!:0" -c:v libx264 -preset slower -crf !CRF_VALUE! -c:a copy "!OUTPUT_DIR!\video_4_tile3.mp4"

echo 生成面4 (左下区域)...
ffmpeg -i "!OUTPUT_CMP!" -vf "crop=!TILE_SIZE!:!TILE_SIZE!:0:!TILE_SIZE!" -c:v libx264 -preset slower -crf !CRF_VALUE! -c:a copy "!OUTPUT_DIR!\video_4_tile4.mp4"

echo 生成面5 (中下区域)...
ffmpeg -i "!OUTPUT_CMP!" -vf "crop=!TILE_SIZE!:!TILE_SIZE!:!TILE_SIZE!:!TILE_SIZE!" -c:v libx264 -preset slower -crf !CRF_VALUE! -c:a copy "!OUTPUT_DIR!\video_4_tile5.mp4"

echo 生成面6 (右下区域)...
set /a X=!TILE_SIZE!*2
ffmpeg -i "!OUTPUT_CMP!" -vf "crop=!TILE_SIZE!:!TILE_SIZE!:!X!:!TILE_SIZE!" -c:v libx264 -preset slower -crf !CRF_VALUE! -c:a copy "!OUTPUT_DIR!\video_4_tile6.mp4"

echo.
echo ========================================
echo 所有24个瓦片已生成完成!
echo.
echo 输出文件夹: !OUTPUT_DIR!
echo.
echo 文件列表:
for %%i in (1 2 3 4 5 6) do (
    echo 面%%i:
    for %%j in (1 2 3 4) do (
        echo   video_4_tile%%i_%%j.mp4
    )
)
echo.
echo 瓦片映射关系:
echo 面1: 左上区域 (前左)
echo 面2: 中上区域 (前右)
echo 面3: 右上区域 (上)
echo 面4: 左下区域 (后左)
echo 面5: 中下区域 (后右)
echo 面6: 右下区域 (下)
echo.
echo 注意: CMP c3x2布局的6个面分配:
echo 列1: 左面
echo 列2: 前面、上面、下面
echo 列3: 右面、后面
echo ========================================
echo.
echo 处理完成! 按任意键退出...
pause >nul