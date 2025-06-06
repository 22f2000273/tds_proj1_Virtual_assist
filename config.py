import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI API (you'll need to get this)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    
    # Discourse settings
    DISCOURSE_BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
    TDS_CATEGORY = "tds"
    
    # Date ranges for scraping
    SCRAPE_START_DATE = "2025-01-01"
    SCRAPE_END_DATE = "2025-04-14"
    
    # Flask settings
    DEBUG = True
    PORT = int(os.getenv('PORT', 5000))