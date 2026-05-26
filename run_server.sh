#!/bin/bash
cd /home/sujeetkumar/Documents/multimodel_chatbot/multimodel_chat_ai/rgChat-backend
export PYTHONPATH=/home/sujeetkumar/Documents/multimodel_chatbot/multimodel_chat_ai/rgChat-backend:$PYTHONPATH
/home/sujeetkumar/Documents/multimodel_chatbot/multimodel_chat_ai/rgChat-backend/venv/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
