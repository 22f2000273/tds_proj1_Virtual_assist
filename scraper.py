import requests
from bs4 import BeautifulSoup
import json
import time
from datetime import datetime, timedelta
from config import Config
import os

class TDSDataScraper:
    def __init__(self):
        self.base_url = Config.DISCOURSE_BASE_URL
        self.session = requests.Session()
        
    def scrape_discourse_posts(self, start_date, end_date):
        """Scrape TDS Discourse posts within date range"""
        posts = []
        
        try:
            # Get TDS category posts
            url = f"{self.base_url}/c/tds.json"
            response = self.session.get(url)
            
            if response.status_code == 200:
                data = response.json()
                topics = data.get('topic_list', {}).get('topics', [])
                
                for topic in topics:
                    # Check if topic is within date range
                    created_at = datetime.strptime(topic['created_at'][:10], '%Y-%m-%d')
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    
                    if start_dt <= created_at <= end_dt:
                        post_data = self.scrape_topic_details(topic['id'])
                        if post_data:
                            posts.append(post_data)
                        time.sleep(1)  # Be respectful
                        
        except Exception as e:
            print(f"Error scraping discourse: {e}")
            
        return posts
    
    def scrape_topic_details(self, topic_id):
        """Get detailed content from a specific topic"""
        try:
            url = f"{self.base_url}/t/{topic_id}.json"
            response = self.session.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract relevant information
                topic_data = {
                    'id': topic_id,
                    'title': data.get('title', ''),
                    'url': f"{self.base_url}/t/{topic_id}",
                    'created_at': data.get('created_at', ''),
                    'posts': []
                }
                
                # Get all posts in the topic
                for post in data.get('post_stream', {}).get('posts', []):
                    topic_data['posts'].append({
                        'content': post.get('cooked', ''),
                        'raw': post.get('raw', ''),
                        'created_at': post.get('created_at', ''),
                        'username': post.get('username', '')
                    })
                
                return topic_data
                
        except Exception as e:
            print(f"Error scraping topic {topic_id}: {e}")
            
        return None
    
    def scrape_course_content(self):
        """Scrape TDS course content - you'll need to adapt this based on available sources"""
        # This is a placeholder - you'll need to identify where course content is available
        course_content = {
            'modules': [],
            'assignments': [],
            'lectures': []
        }
        
        # Example: If course content is available on Discourse or other platforms
        # Add your scraping logic here
        
        return course_content
    
    def save_data(self):
        """Save scraped data to files"""
        os.makedirs('data', exist_ok=True)
        
        # Scrape discourse posts
        print("Scraping Discourse posts...")
        discourse_posts = self.scrape_discourse_posts(
            Config.SCRAPE_START_DATE, 
            Config.SCRAPE_END_DATE
        )
        
        with open('data/discourse_posts.json', 'w') as f:
            json.dump(discourse_posts, f, indent=2)
        
        # Scrape course content
        print("Scraping course content...")
        course_content = self.scrape_course_content()
        
        with open('data/course_content.json', 'w') as f:
            json.dump(course_content, f, indent=2)
        
        print(f"Scraped {len(discourse_posts)} discourse posts")
        print("Data saved successfully!")

if __name__ == "__main__":
    scraper = TDSDataScraper()
    scraper.save_data()