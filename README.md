# TDS Virtual TA

A Virtual Teaching Assistant for IIT Madras Tools in Data Science course that automatically responds to student questions based on course content and Discourse discussions.

## Features

- Automatic question answering using AI
- Support for image attachments
- Integration with TDS Discourse forum data
- REST API for easy integration
- Scraped data from course materials and discussions

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables (OpenAI API key)
4. Run data scraping: `python scraper.py`
5. Start the API: `python app.py`

## API Usage

POST to `/api/` with JSON:
```json
"""{
  "question": "Your question here",
  "image": "base64_encoded_image (optional)"
}
"""
