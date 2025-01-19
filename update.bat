@echo off

call :colored Updating... Green
git pull
echo ---

@pause

:colored
%Windir%\System32\WindowsPowerShell\v1.0\Powershell.exe write-host -foregroundcolor %2 %1
