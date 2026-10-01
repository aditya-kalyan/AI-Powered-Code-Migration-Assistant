$ErrorActionPreference = "Stop"

if (-not $env:OPENAI_API_KEY) {
    Write-Host "OPENAI_API_KEY is not set."
    Write-Host "Set it with:"
    Write-Host "  `$env:OPENAI_API_KEY = 'sk-your-key-here'"
    Write-Host "Then run this script again."
    exit 1
}

python -m pip install -r requirements.txt
python app.py
