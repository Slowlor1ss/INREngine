@echo off
echo Launching comparison instances...

:: Instance 1: Standard (No Positional Encoding)
start "Standard Network (NO PE)" x64\Release\3b1b_backprop.exe --set-live 1 --i laurens.bmp --o OUT/weights_biases_Laurens_BASE.csv --set-pe 0 --batch 32 --lr 0.25

:: Instance 2: With Positional Encoding
start "PE Network (WITH PE)" x64\Release\3b1b_backprop.exe --set-live 1 --i laurens.bmp --o OUT/weights_biases_Laurens_PE.csv --set-pe 1 --freq 7 --batch 32 --lr 0.25

echo Comparison instances launched!
pause