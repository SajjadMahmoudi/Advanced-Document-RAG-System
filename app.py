from flask import Flask, request, jsonify, render_template, Response, stream_with_context
import os
from werkzeug.utils import secure_filename
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.callbacks.base import BaseCallbackHandler
import tempfile
import json
import requests
import threading
import base64
from flask_socketio import SocketIO, emit
from urllib.parse import urlparse
from TTS import StreamingTTSProcessor
import io
import wave

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Supported file extensions
ALLOWED_EXTENSIONS = {
    'pdf', 'docx', 'doc', 'pptx', 'ppt', 
    'xlsx', 'xls', 'txt', 'md', 'html', 
    'rtf', 'odt', 'csv'
}

# Available models
AVAILABLE_MODELS = {
    'qwen3:0.6b': 'Qwen 2.5 0.6B (Fast)',
    'qwen3-vl:2b': 'Qwen 2.5 VL 2B (Vision+Text)',
    'llava:7b': 'LLaVA 7B (Vision+Text)',
    'alibo/dorna:8b-instruct-q5': 'Dorna 8B Instruct (High Quality)'
}

# Global variables
vectorstore = None
qa_chain = None
document_processed = False
current_model = 'qwen3:0.6b'

# Language code mapping for TTS
LANGUAGE_TO_TTS = {
    'english': 'b',  # British English
    'american_english': 'a',
    'spanish': 'e',
    'french': 'f',
    'hindi': 'h',
    'italian': 'i',
    'portuguese': 'p',
    'japanese': 'j',
    'chinese': 'z',
    'persian': 'b',  # Use English for Persian (no Persian support in Kokoro)
    'german': 'b',   # Use English for German
    'arabic': 'b',   # Use English for Arabic
    'russian': 'b',  # Use English for Russian
    'korean': 'b'    # Use English for Korean
}

# Custom callback handler for SSE streaming
class SSECallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
        
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_document(file_path_or_url):
    """Extract text from various document formats using docling"""
    try:
        converter = DocumentConverter()
        
        # Check if it's a URL
        if isinstance(file_path_or_url, str) and file_path_or_url.startswith(('http://', 'https://')):
            print(f"Processing URL directly: {file_path_or_url}")
            result = converter.convert(file_path_or_url)
        else:
            result = converter.convert(file_path_or_url)
        
        markdown_text = result.document.export_to_markdown()
        return markdown_text
        
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        
        if isinstance(file_path_or_url, str) and file_path_or_url.startswith(('http://', 'https://')):
            print("Trying fallback download method...")
            try:
                response = requests.get(file_path_or_url, timeout=30)
                response.raise_for_status()
                
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False, 
                    suffix='.tmp',
                    dir=app.config['UPLOAD_FOLDER']
                )
                temp_file.write(response.content)
                temp_file.close()
                
                result = converter.convert(temp_file.name)
                markdown_text = result.document.export_to_markdown()
                
                os.remove(temp_file.name)
                return markdown_text
                
            except Exception as download_error:
                print(f"Fallback download also failed: {str(download_error)}")
        
        if isinstance(file_path_or_url, str) and (file_path_or_url.endswith('.txt') or file_path_or_url.endswith('.md')):
            if file_path_or_url.startswith(('http://', 'https://')):
                response = requests.get(file_path_or_url, timeout=30)
                return response.text
            else:
                with open(file_path_or_url, 'r', encoding='utf-8') as f:
                    return f.read()
        
        raise

def process_document(file_path_or_url, model_name=None):
    """Process document and create vector store"""
    global vectorstore, qa_chain, document_processed, current_model
    
    if model_name:
        current_model = model_name
    
    print(f"Extracting text from: {file_path_or_url}")
    text_content = extract_text_from_document(file_path_or_url)
    
    if not text_content.strip():
        raise ValueError("No text could be extracted from the document")
    
    print(f"Extracted {len(text_content)} characters")

    # Store extracted text globally
    app.extracted_text_content = text_content
    
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text_content)
    print(f"Created {len(chunks)} chunks")
    
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small"
    )
    
    print("Building vector store...")
    vectorstore = Chroma.from_texts(chunks, embeddings)
    
    print(f"Initializing LLM: {current_model}...")
    try:
        llm = Ollama(
            model=current_model,
            temperature=0.7,
            base_url="http://localhost:11434",
            timeout=120
        )
        
        response = llm.invoke("hello")
        print(f"Success! Response: {response[:50]}...")
        
    except Exception as e:
        print(f"Error with model {current_model}: {e}")
        raise Exception(f"Ollama model {current_model} failed. Please run: ollama pull {current_model}")
    
    print("Creating QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    document_processed = True
    print(f"Document processing complete with model: {current_model}")
    return True

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', models=AVAILABLE_MODELS)

@app.route('/models', methods=['GET'])
def get_models():
    """Get available models"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'current_model': current_model
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle document upload"""
    global document_processed
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not supported. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    selected_model = request.form.get('model', 'qwen3:0.6b')
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"Processing document: {filepath} with model: {selected_model}")
        
        process_document(filepath, selected_model)
        
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Document processed successfully with {AVAILABLE_MODELS.get(selected_model, selected_model)}. You can now ask questions!',
            'current_model': selected_model,
            'filename': filename
        })
    
    except Exception as e:
        document_processed = False
        print(f"Error processing document: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing document: {str(e)}'}), 500
    
@app.route('/upload_url', methods=['POST'])
def upload_from_url():
    """Handle document upload from URL"""
    global document_processed
    
    data = request.get_json()
    url = data.get('url', '').strip()
    
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    
    try:
        parsed = urlparse(url)
        if not parsed.scheme or parsed.scheme not in ['http', 'https']:
            return jsonify({'error': 'Invalid URL scheme. Use http:// or https://'}), 400
    except:
        return jsonify({'error': 'Invalid URL format'}), 400
    
    selected_model = data.get('model', 'qwen3:0.6b')
    
    try:
        print(f"Processing document from URL: {url} with model: {selected_model}")
        
        process_document(url, selected_model)
        
        domain = urlparse(url).netloc
        
        return jsonify({
            'success': True,
            'message': f'Document from {domain} processed successfully with {AVAILABLE_MODELS.get(selected_model, selected_model)}',
            'current_model': selected_model,
            'filename': f"URL: {domain}",
            'source': 'url'
        })
    
    except Exception as e:
        document_processed = False
        print(f"Error processing document from URL: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing document from URL: {str(e)}'}), 500
    
@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle question answering (non-streaming)"""
    global qa_chain, document_processed, current_model
    
    if not document_processed or qa_chain is None:
        return jsonify({'error': 'Please upload a document first'}), 400
    
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Question cannot be empty'}), 400
    
    try:
        print(f"Answering question with model {current_model}: {question}")
        
        result = qa_chain.invoke({"query": question})
        
        answer = result['result']
        sources = result.get('source_documents', [])
        
        print(f"Answer: {answer[:100]}...")
        
        return jsonify({
            'success': True,
            'answer': answer,
            'sources': [doc.page_content[:200] + '...' for doc in sources[:2]],
            'model_used': current_model
        })
    
    except Exception as e:
        print(f"Error answering question: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error answering question: {str(e)}'}), 500

@app.route('/ask_stream', methods=['POST'])
def ask_question_stream():
    """Handle question answering with SSE streaming"""
    global qa_chain, document_processed, current_model, vectorstore
    
    if not document_processed or qa_chain is None:
        return jsonify({'error': 'Please upload a document first'}), 400
    
    data = request.get_json()
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Question cannot be empty'}), 400
    
    def generate():
        try:
            print(f"Streaming answer with model {current_model}: {question}")
            
            docs = vectorstore.similarity_search(question, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            streaming_llm = Ollama(
                model=current_model,
                temperature=0.7,
                base_url="http://localhost:11434",
                timeout=120
            )
            
            prompt = f"""Based on the following context, answer the question. If you cannot find the answer in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
            
            full_answer = ""
            for chunk in streaming_llm.stream(prompt):
                if chunk:
                    full_answer += chunk
                    yield f"data: {json.dumps({'token': chunk, 'type': 'token'})}\n\n"
            
            sources = [doc.page_content[:200] + '...' for doc in docs[:2]]
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources, 'model_used': current_model})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            print(f"Error in streaming: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/change_model', methods=['POST'])
def change_model():
    """Change the current model"""
    global current_model, qa_chain, vectorstore
    
    data = request.get_json()
    new_model = data.get('model', '').strip()
    
    if not new_model:
        return jsonify({'error': 'Model name is required'}), 400
    
    if new_model not in AVAILABLE_MODELS:
        return jsonify({'error': f'Model {new_model} is not available'}), 400
    
    try:
        print(f"Testing new model: {new_model}")
        test_llm = Ollama(
            model=new_model,
            base_url="http://localhost:11434",
            timeout=60
        )
        test_response = test_llm.invoke("hello")
        print(f"Model test successful: {test_response[:50]}...")
        
        old_model = current_model
        current_model = new_model
        
        if vectorstore is not None:
            llm = Ollama(
                model=current_model,
                temperature=0.7,
                base_url="http://localhost:11434",
                timeout=120
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True
            )
        
        return jsonify({
            'success': True,
            'message': f'Model changed from {old_model} to {new_model}',
            'current_model': current_model,
            'model_display_name': AVAILABLE_MODELS.get(current_model, current_model)
        })
        
    except Exception as e:
        print(f"Error changing model: {str(e)}")
        return jsonify({'error': f'Failed to change model: {str(e)}'}), 500
    
@app.route('/translate', methods=['POST'])
def translate_text():
    """Translate text to a specific language"""
    data = request.get_json()
    text = data.get('text', '').strip()
    language = data.get('language', '').strip()
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    if not language or language == 'none':
        return jsonify({'error': 'No language selected'}), 400
    
    language_prompts = {
        'persian': 'Persian (فارسی)',
        'spanish': 'Spanish (Español)',
        'french': 'French (Français)',
        'chinese': 'Chinese (中文)',
        'german': 'German (Deutsch)',
        'japanese': 'Japanese (日本語)',
        'arabic': 'Arabic (العربية)',
        'russian': 'Russian (Русский)',
        'portuguese': 'Portuguese (Português)',
        'italian': 'Italian (Italiano)',
        'korean': 'Korean (한국어)',
        'hindi': 'Hindi (हिन्दी)'
    }
    
    if language not in language_prompts:
        return jsonify({'error': f'Unsupported language: {language}'}), 400
    
    try:
        translation_llm = Ollama(
            model='alibo/dorna:8b-instruct-q5',
            temperature=0.3,
            base_url="http://localhost:11434",
            timeout=60
        )
        
        target_language = language_prompts[language]
        prompt = f"""Translate the following text to {target_language}. 
Keep the meaning accurate and maintain any technical terms or proper names as they are.
Only output the translated text, no explanations or additional text.

Text to translate:
{text}

Translation in {target_language}:"""
        
        print(f"Translating text to {language}...")
        translated_text = translation_llm.invoke(prompt)
        translated_text = translated_text.strip()
        
        return jsonify({
            'success': True,
            'translated_text': translated_text,
            'original_length': len(text),
            'translated_length': len(translated_text),
            'language': language
        })
        
    except Exception as e:
        print(f"Error in translation: {str(e)}")
        return jsonify({'error': f'Translation failed: {str(e)}'}), 500

@app.route('/reset', methods=['POST'])
def reset():
    """Reset the system"""
    global vectorstore, qa_chain, document_processed, current_model
    
    vectorstore = None
    qa_chain = None
    document_processed = False
    current_model = 'qwen3:0.6b'
    
    print("System reset")
    return jsonify({
        'success': True, 
        'message': 'System reset successfully',
        'current_model': current_model
    })

def numpy_to_wav_bytes(audio_array, sample_rate=24000):
    """Convert numpy array to WAV bytes"""
    byte_io = io.BytesIO()
    
    with wave.open(byte_io, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes = 16 bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array.tobytes())
    
    byte_io.seek(0)
    return byte_io.read()

@socketio.on('start_tts')
def handle_start_tts(data):
    """Handle TTS streaming request via WebSocket"""
    try:
        text = data.get('text', '').strip()
        language = data.get('language', 'english').lower()
        
        print(f"\n{'='*60}")
        print(f"Received TTS request from client: {request.sid}")
        print(f"Language: {language}")
        print(f"Text length: {len(text)}")
        print(f"{'='*60}\n")
        
        if not text:
            print("ERROR: No text provided")
            emit('tts_error', {'error': 'No text provided'})
            return
        
        # Get language code for TTS
        lang_code = LANGUAGE_TO_TTS.get(language, 'b')
        print(f"Mapped language '{language}' to TTS code: '{lang_code}'")
        
        # Start TTS in background thread
        print("Starting background TTS thread...")
        threading.Thread(
            target=stream_tts_to_client,
            args=(text, lang_code, request.sid),
            daemon=True
        ).start()
        
        emit('tts_started', {'message': 'TTS generation started'})
        print("TTS thread started, confirmation sent to client\n")
        
    except Exception as e:
        print(f"ERROR in handle_start_tts: {str(e)}")
        import traceback
        traceback.print_exc()
        emit('tts_error', {'error': str(e)})

def stream_tts_to_client(text: str, lang_code: str, client_sid: str):
    """Stream TTS audio chunks to client via WebSocket"""
    try:
        print(f"\n{'='*60}")
        print(f"Starting TTS stream for client: {client_sid}")
        print(f"Language: {lang_code}")
        print(f"Text length: {len(text)} characters")
        print(f"Text preview: {text[:100]}...")
        print(f"{'='*60}\n")
        
        tts = StreamingTTSProcessor(lang_code=lang_code)
        
        chunk_count = 0
        for audio_chunk in tts.stream_speech(text):
            print(f"\nProcessing audio chunk {chunk_count}")
            print(f"  - Shape: {audio_chunk.shape}")
            print(f"  - Dtype: {audio_chunk.dtype}")
            print(f"  - Min: {audio_chunk.min()}, Max: {audio_chunk.max()}")
            
            # Convert numpy array to WAV bytes using the TTS instance method
            wav_bytes = tts.numpy_to_wav_bytes(audio_chunk)
            print(f"  - WAV size: {len(wav_bytes)} bytes")
            
            # Convert to base64 for transmission
            audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
            print(f"  - Base64 size: {len(audio_b64)} characters")
            
            # Emit via SocketIO to specific client
            socketio.emit('tts_audio_chunk', {
                'chunk_index': chunk_count,
                'audio_data': audio_b64,
                'sample_rate': tts.sample_rate
            }, room=client_sid)
            
            print(f"  - Chunk {chunk_count} sent successfully ✓")
            chunk_count += 1
        
        # Emit completion
        socketio.emit('tts_complete', {
            'total_chunks': chunk_count,
            'message': 'TTS generation complete'
        }, room=client_sid)
        
        print(f"\n{'='*60}")
        print("TTS completed successfully!")
        print(f"Total chunks sent: {chunk_count}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR in TTS streaming: {str(e)}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        socketio.emit('tts_error', {
            'error': str(e)
        }, room=client_sid)

@socketio.on('connect')
def handle_connect():
    print(f'\n{"="*60}')
    print(f'Client connected: {request.sid}')
    print(f'{"="*60}\n')
    emit('connected', {'message': 'Connected to TTS server'})

@socketio.on('disconnect')
def handle_disconnect():
    print(f'\n{"="*60}')
    print(f'Client disconnected: {request.sid}')
    print(f'{"="*60}\n')

@app.route('/get_extracted_text', methods=['GET'])
def get_extracted_text():
    """Get the extracted text from the last processed document"""
    global extracted_text_content
    
    if not hasattr(app, 'extracted_text_content') or not app.extracted_text_content:
        return jsonify({'error': 'No document has been processed yet'}), 400
    
    return jsonify({
        'success': True,
        'extracted_text': app.extracted_text_content,
        'length': len(app.extracted_text_content)
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Flask-SocketIO server...")
    print("TTS Feature: ENABLED")
    print("WebSocket: ENABLED")
    print("="*60 + "\n")
    socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)