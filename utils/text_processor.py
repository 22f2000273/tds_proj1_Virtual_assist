import re
import json
from typing import List, Dict

class TextProcessor:
    def __init__(self):
        self.load_data()
    
    def load_data(self):
        """Load scraped data"""
        try:
            with open('data/discourse_posts.json', 'r') as f:
                self.discourse_data = json.load(f)
        except FileNotFoundError:
            self.discourse_data = []
            
        try:
            with open('data/course_content.json', 'r') as f:
                self.course_data = json.load(f)
        except FileNotFoundError:
            self.course_data = {}
    
    def search_relevant_content(self, question: str) -> List[Dict]:
        """Search for relevant content based on question"""
        relevant_posts = []
        question_lower = question.lower()
        
        # Search through discourse posts
        for post in self.discourse_data:
            if self.is_relevant(post, question_lower):
                relevant_posts.append({
                    'title': post.get('title', ''),
                    'url': post.get('url', ''),
                    'content': self.extract_relevant_text(post, question_lower),
                    'type': 'discourse'
                })
        
        return relevant_posts[:5]  # Return top 5 relevant posts
    
    def is_relevant(self, post: Dict, question: str) -> bool:
        """Check if post is relevant to question"""
        # Simple keyword matching - you can improve this
        keywords = self.extract_keywords(question)
        
        text_to_search = (
            post.get('title', '').lower() + ' ' +
            ' '.join([p.get('raw', '') for p in post.get('posts', [])])
        ).lower()
        
        return any(keyword in text_to_search for keyword in keywords)
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Remove common words and extract meaningful terms
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'should', 'would', 'could', 'can', 'will', 'i', 'you', 'we', 'they', 'he', 'she', 'it'}
        
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in common_words and len(word) > 2]
        
        return keywords
    
    def extract_relevant_text(self, post: Dict, question: str) -> str:
        """Extract most relevant text from post"""
        # Get first post content as most relevant
        if post.get('posts'):
            return post['posts'][0].get('raw', '')[:500]  # First 500 chars
        return ''