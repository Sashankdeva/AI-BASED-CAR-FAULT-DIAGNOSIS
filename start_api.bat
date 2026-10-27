@echo off
cd /d "%~dp0"
if not exist models\fault_model.joblib python -m car_fault.train
echo Dashboard: http://localhost:8005/ui/   API docs: http://localhost:8005/docs
python -m uvicorn api.main:app --host 0.0.0.0 --port 8005 --reload
