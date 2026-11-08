@echo off
cd /d %~dp0
if not exist .venv (
  py -3 -m venv .venv || python -m venv .venv
  call .venv\Scripts\activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate
  pip install -q starlette uvicorn
)
echo.
echo  Open on your iPhone (same Wi-Fi):  http://YOUR-PC-IP:8000
echo  Find YOUR-PC-IP with: ipconfig  (IPv4 Address)
echo.
uvicorn server:app --host 0.0.0.0 --port 8000
