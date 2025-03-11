#!/bin/bash

# Start the backend server
echo "Starting backend server..."
cd backend
python main.py &
BACKEND_PID=$!

# Start the frontend development server
echo "Starting frontend development server..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!

# Function to handle script termination
function cleanup {
  echo "Stopping servers..."
  kill $BACKEND_PID
  kill $FRONTEND_PID
  exit
}

# Trap SIGINT (Ctrl+C) and call cleanup
trap cleanup SIGINT

# Keep the script running
echo "Both servers are running. Press Ctrl+C to stop."
wait 