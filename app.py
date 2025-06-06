from flask import Flask, request, jsonify
import base64
import json
from PIL import Image
import io
from utils.text_processor import TextProcessor
from utils.answer_generator import AnswerGenerator
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize components
text_processor = TextProcessor()
answer_generator = AnswerGenerator()

@app.route('/api/', methods=['POST'])
def handle_question():
    """Main API endpoint to handle student questions"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({'error': 'Question is required'}), 400
        
        question = data['question']
        image_data = data.get('image')
        
        # Process image if provided
        image_info = None
        if image_data:
            image_info = process_image(image_data)
        
        # Find relevant content
        relevant_content = text_processor.search_relevant_content(question)
        
        # Generate answer
        response = answer_generator.generate_answer(question, relevant_content, image_info)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': 'An error occurred processing your request',
            'answer': 'I apologize for the technical issue. Please post your question on the TDS Discourse forum.',
            'links': [
                {
                    'url': 'https://discourse.onlinedegree.iitm.ac.in/c/tds',
                    'text': 'TDS Discourse Forum'
                }
            ]
        }), 500

def process_image(base64_image):
    """Process base64 encoded image"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        
        # Basic image info
        return {
            'format': image.format,
            'size': image.size,
            'mode': image.mode
        }
    except Exception as e:
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'TDS Virtual TA is running'})

@app.route('/', methods=['GET'])
def home():
    """Home page with API documentation"""
    return jsonify({
        'message': 'TDS Virtual TA API',
        'endpoints': {
            'POST /api/': 'Submit questions with optional image attachments',
            'GET /health': 'Health check'
        },
        'example_request': {
            'question': 'Should I use gpt-4o-mini or gpt-3.5-turbo?',
            'image': 'base64_encoded_image_data (optional)'
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=Config.PORT, debug=Config.DEBUG)