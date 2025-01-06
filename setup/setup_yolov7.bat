@echo off
echo Choose an option:
echo 1. Standard YOLO (WongKinYiu/yolov7)
echo 2. Modified YOLO (GitJvG/yolov7)
set /p choice="Enter 1 for Standard YOLOv7 or 2 for Modified YOLOv7: "

if "%choice%"=="1" (
    echo Cloning standard YOLO repository from https://github.com/WongKinYiu/yolov7...
    git clone https://github.com/WongKinYiu/yolov7
    cd yolov7
    git checkout 44f30af0daccb1a3baecc5d80eae22948516c579
    cd ..
) else if "%choice%"=="2" (
    echo Cloning modified YOLO repository from https://github.com/GitJvG/yolov7...
    git clone https://github.com/GitJvG/yolov7
) else (
    echo Invalid choice. Please run the script again and enter either 1 or 2.
)

echo Creating virtual environment in '.venv'...
python -m venv .venv

echo Installing requirements from requirements.txt...
.venv\Scripts\pip install -r yolov7/seg/requirements.txt

echo Installing ipykernel...
.venv\Scripts\pip install ipykernel

echo Setting up Jupyter kernel...
.venv\Scripts\python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"

echo All tasks completed successfully!
pause
