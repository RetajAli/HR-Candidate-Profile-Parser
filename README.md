# HR Candidate Profile Parser

An AI-powered web application that extracts and organizes candidate information from resumes into structured JSON.

The application was built using Python, Streamlit, LangChain, and the Hugging Face Inference API.

## Features

- Upload PDF, DOCX, and TXT resumes
- AI-powered candidate information extraction
- Extract candidate name and email
- Extract education, skills, and work experience
- Display a candidate profile dashboard
- Calculate profile completeness
- Detect missing information
- View structured JSON output
- Download candidate information as JSON

## Technologies Used

- Python
- Streamlit
- LangChain
- Hugging Face Inference API
- Pydantic
- PyMuPDF
- python-docx
- Pandas

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/HR-Candidate-Profile-Parser.git
```

Open the project folder:

```bash
cd HR-Candidate-Profile-Parser
```

Install the requirements:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```text
HF_TOKEN=your_huggingface_token
```

Run the application:

```bash
streamlit run hr_parser.py
```

## Supported Resume Formats

- PDF
- DOCX
- TXT

## Important Note

The `.env` file is not included because it contains a private Hugging Face API token.

## Author

Retaj Ali Ismael
