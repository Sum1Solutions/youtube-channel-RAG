from flask import Flask, render_template, request, jsonify, session
import os
from dotenv import load_dotenv
from youtube_extractor import YouTubeExtractor
from rag_system import RAGSystem
import json

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Initialize systems
youtube_extractor = YouTubeExtractor()
rag_system = RAGSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/preview_content', methods=['POST'])
def preview_content():
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'YouTube URL is required'}), 400
        
        # Detect URL type
        url_type = youtube_extractor.detect_url_type(url)
        
        # Get content videos without extracting transcripts
        result = youtube_extractor.get_content_videos(url)
        
        if result['success']:
            return jsonify({
                'success': True,
                'url_type': url_type,
                'content_type': result['content_type'],
                'title': result['title'],
                'total_videos': len(result['videos']),
                'videos': result['videos'][:5]  # Show first 5 videos as preview
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/extract_content', methods=['POST'])
def extract_content():
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'error': 'YouTube URL is required'}), 400
        
        # Extract content videos and transcripts
        result = youtube_extractor.extract_content_transcripts(url)
        
        if result['success']:
            # Store transcripts in RAG system
            rag_system.add_transcripts(result['transcripts'])
            
            # Store content info in session
            session['content_info'] = {
                'content_type': result['content_type'],
                'title': result['title'],
                'video_count': len(result['transcripts']),
                'total_videos': result['total_videos']
            }
            
            content_type_name = {
                'video': 'video',
                'playlist': 'playlist', 
                'channel': 'channel'
            }.get(result['content_type'], 'content')
            
            return jsonify({
                'success': True,
                'message': f"Successfully extracted {len(result['transcripts'])} video transcripts out of {result['total_videos']} total videos from {content_type_name}",
                'content_type': result['content_type'],
                'title': result['title'],
                'video_count': len(result['transcripts']),
                'total_videos': result['total_videos'],
                'successful_transcripts': len(result['transcripts'])
            })
        else:
            return jsonify({'error': result['error']}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Search in RAG system
        results = rag_system.search(query, top_k=5)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        question = data.get('question')
        
        if not question:
            return jsonify({'error': 'Question is required'}), 400
        
        # Generate answer using RAG
        answer = rag_system.generate_answer(question)
        
        return jsonify({
            'success': True,
            'answer': answer
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    content_info = session.get('content_info', {})
    return jsonify({
        'has_data': bool(content_info),
        'content_info': content_info
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)