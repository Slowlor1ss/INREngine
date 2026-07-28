@echo off
:: Set the number of instances you want to run here
set INSTANCES=3

echo Launching %INSTANCES% instances of the network...

:: Loop from 1 to INSTANCES, stepping by 1
FOR /L %%i IN (1, 1, %INSTANCES%) DO (
    
    :: The text in quotes sets the title of the console window
    :: Add your shared settings at the end of this line
    start "Instance %%i" 3b1b_backprop.exe --batch 64 --lr 0.25 --use-pe --freq 7
    
)

echo All instances launched!
pause