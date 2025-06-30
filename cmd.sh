#!/bin/bash

echo "Starting main app..."

if gunicorn api.index:app --bind 0.0.0.0:8000; then
    echo "App ran successfully."
else
    echo "App crashed, trying again..."
    gunicorn index:app --bind 0.0.0.0:8000
fi



