# Start the Streamlit GUI for Arabic Sign Language Recognition
Write-Host "Starting Arabic Sign Language Recognition GUI..." -ForegroundColor Green
Write-Host "Using virtual environment..." -ForegroundColor Cyan
Write-Host "The application will open in your default web browser." -ForegroundColor Cyan
Write-Host ""

# Run Streamlit using the virtual environment's Python

first --> Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
second --> & "D:\Natiq project proposal\.venv\Scripts\Activate.ps1"
third --> & .\.venv\Scripts\streamlit.exe run gui_app.py --server.headless true
 