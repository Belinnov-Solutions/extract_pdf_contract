# Contract OCR Service

A FastAPI-based service that extracts structured information from PDF contracts using OCR technology.

## Features

- Extract customer information (name, phone, address)
- Extract device details (model, IMEI, SIM number)
- Extract plan information (name, charges, dates)
- Extract contract dates and order numbers
- RESTful API with automatic documentation

## Prerequisites

- Python 3.8+
- Tesseract OCR Engine

## Installation

1. **Install Python 3.8+** from [python.org](https://www.python.org/downloads/)

2. **Install Tesseract OCR:**

   - Windows: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
   - Install to: `C:\Program Files\Tesseract-OCR\`

3. **Clone the repository:**

   ```bash
   git clone <your-repo-url>
   cd contract-ocr-service
   ```

4. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

```bash
python -m uvicorn app.main:app --reload
```

Or use the provided run script:

```bash
python run.py
```

## API Endpoints

- **Health Check:** `GET /health`
- **Extract Contract:** `POST /extract`
- **API Documentation:** `http://localhost:8000/docs`

## Usage

1. Start the server
2. Go to `http://localhost:8000/docs`
3. Use the `/extract` endpoint to upload a PDF contract
4. Receive structured JSON data with extracted information

## Environment Variables

- `TESSERACT_CMD`: Path to Tesseract executable (optional, defaults to standard Windows path)

## Project Structure

```
app/
├── __init__.py
├── main.py          # FastAPI application
├── models.py        # Pydantic data models
└── ocr.py          # OCR processing logic
```
