import openai
from typing import List, Dict
from config import Config

class AnswerGenerator:
    def __init__(self):
        openai.api_key = Config.OPENAI_API_KEY
    
    def generate_answer(self, question: str, relevant_content: List[Dict], image_data: str = None) -> Dict:
        """Generate answer using OpenAI GPT"""
        
        # Prepare content for GPT
        context = self.prepare_context(relevant_content)
        
        # Create prompt
        prompt = f"""
You are a Teaching Assistant for the Tools in Data Science course at IIT Madras.
Answer the student's question based on the provided context from course materials and Discourse discussions.

Context from course materials and discussions:
{context}

Student Question: {question}

Please provide:
1. A clear, helpful answer
2. Reference the most relevant sources

Answer in a conversational, helpful tone as a TA would.
"""

        try:
            # Use GPT-3.5-turbo for cost efficiency
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful Teaching Assistant for Tools in Data Science course."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content
            
            # Prepare links from relevant content
            links = []
            for content in relevant_content[:3]:  # Top 3 links
                if content.get('url'):
                    links.append({
                        "url": content['url'],
                        "text": content.get('title', 'Relevant discussion')[:100]
                    })
            
            return {
                "answer": answer,
                "links": links
            }
            
        except Exception as e:
            # Fallback response
            return {
                "answer": "I apologize, but I'm having trouble generating a response right now. Please try asking your question in the Discourse forum where TAs and fellow students can help you.",
                "links": [
                    {
                        "url": "https://discourse.onlinedegree.iitm.ac.in/c/tds",
                        "text": "TDS Discourse Forum"
                    }
                ]
            }
    
    def prepare_context(self, relevant_content: List[Dict]) -> str:
        """Prepare context string from relevant content"""
        context_parts = []
        
        for i, content in enumerate(relevant_content[:3], 1):
            context_parts.append(f"Source {i}: {content.get('title', 'Discussion')}")
            context_parts.append(f"Content: {content.get('content', '')[:300]}...")
            context_parts.append("---")
        
        return "\n".join(context_parts)