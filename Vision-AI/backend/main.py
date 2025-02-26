from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import shutil
import uuid
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import requests
import base64
from groq import Groq
from dotenv import load_dotenv



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

USE_LOCAL_EMBEDDINGS = True 


try:
    if USE_LOCAL_EMBEDDINGS:
        from sentence_transformers import SentenceTransformer
        print("Loading local embedding model...")
        # Using a smaller model for efficiency - you can change to a larger one if needed
        local_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print(f"Local embedding model loaded: all-MiniLM-L6-v2 (embedding size: {local_embedding_model.get_sentence_embedding_dimension()})")
        EMBEDDING_SIZE = local_embedding_model.get_sentence_embedding_dimension()
    else:
        EMBEDDING_SIZE = 1024  # Cohere's embedding size
except ImportError:
    print("sentence-transformers not installed. Using random embeddings.")
    print("To install: pip install sentence-transformers")
    USE_LOCAL_EMBEDDINGS = False
    EMBEDDING_SIZE = 1024
except Exception as e:
    print(f"Error loading local embedding model: {e}")
    USE_LOCAL_EMBEDDINGS = False
    EMBEDDING_SIZE = 1024

try:
    WHISPER_AVAILABLE = True
    print("Whisper speech recognition is available")
    
    # Models: tiny, base, small, medium, large
    import whisper
    whisper_model = whisper.load_model("tiny")
except ImportError:
    WHISPER_AVAILABLE = False
    print("Whisper is not available. Install with: pip install faster-whisper")

# Fallback to traditional speech recognition if Whisper isn't available
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
    print("Traditional speech recognition is available as fallback")
    
    # Initialize the recognizer
    speech_recognizer = sr.Recognizer()
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("Speech recognition fallback is not available. Install with: pip install SpeechRecognition")

# Configure storage directories
UPLOAD_DIR = Path("uploads")
INDEX_DIR = Path("index")
VECTOR_DIR = Path("vectordb")

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)
# Path to stored vectors and metadata
VECTORS_FILE = VECTOR_DIR / "vectors.npz"
METADATA_FILE = INDEX_DIR / "metadata.json"
CHUNKS_FILE = INDEX_DIR / "chunks.json"

# Initialize global variables with proper types
document_chunks = []
document_vectors = np.zeros((0, EMBEDDING_SIZE), dtype=np.float32)  # Updated to use dynamic embedding size
document_metadata = {}

# Load existing data if available
try:
    if CHUNKS_FILE.exists() and METADATA_FILE.exists() and VECTORS_FILE.exists():
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            document_chunks = json.load(f)
            
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            document_metadata = json.load(f)
            
        loaded_data = np.load(VECTORS_FILE)
        document_vectors = loaded_data["vectors"]
        
        print(f"Loaded {len(document_chunks)} document chunks and metadata")
except Exception as e:
    print(f"Error loading existing data: {e}")
    document_chunks = []
    document_metadata = {}
    document_vectors = np.zeros((0, 1024), dtype=np.float32)  # Fixed size for empty array


# Request models
class ChatRequest(BaseModel):
    message: str
    context_chain: Optional[List[Dict[str, str]]] = None  # Previous messages in the chain
    context_window: int = 5  # How many previous exchanges to keep
    attachments: Optional[List[str]] = None


class DocumentResponse(BaseModel):
    id: str
    filename: str
    upload_date: str
    doc_type: str


class AudioTranscriptionRequest(BaseModel):
    """Request model for audio transcription."""
    filename: Optional[str] = None


@app.post("/transcribe_audio")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    request_data: Optional[str] = Form(None)
):
    """
    Transcribe speech from an audio file using local Whisper model.
    Falls back to traditional speech recognition if Whisper is not available.
    """
    if not WHISPER_AVAILABLE and not SPEECH_RECOGNITION_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Speech recognition is not available. Install required packages."
        )
    
    try:
        # Parse request data if provided
        request_info = {}
        if request_data:
            try:
                request_info = json.loads(request_data)
            except:
                pass
        
        # Save the uploaded audio file temporarily
        temp_audio_path = Path("temp_audio")
        temp_audio_path.mkdir(exist_ok=True)
        
        temp_file_path = temp_audio_path / f"temp_{uuid.uuid4()}.wav"
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        print(f"Audio file saved temporarily at {temp_file_path}")
        
        if WHISPER_AVAILABLE:
            try:
                print("Transcribing with Whisper model...")
                segments = whisper_model.transcribe(str(temp_file_path))
                
                # Collect all segments into complete transcription
                result_text = segments["text"]
                
                result_text = result_text.strip()
                print(f"Whisper transcription: {result_text}")
                
                # Clean up the temporary file
                try:
                    os.remove(temp_file_path)
                except:
                    pass
                    
                return {"text": result_text or "No speech detected"}
            except Exception as whisper_error:
                print(f"Error in Whisper transcription: {whisper_error}")
                # Fall through to traditional speech recognition
        
        if SPEECH_RECOGNITION_AVAILABLE:
            try:
                with sr.AudioFile(str(temp_file_path)) as source:
                    # Record the audio data
                    audio_data = speech_recognizer.record(source)
                    
                    # Try to recognize using Google's API
                    try:
                        print("Attempting with Google Speech Recognition...")
                        text = speech_recognizer.recognize_google(audio_data)
                        print(f"Google Speech Recognition result: {text}")
                    except sr.RequestError:
                        # Try offline Sphinx as last resort
                        try:
                            print("Trying offline PocketSphinx...")
                            text = speech_recognizer.recognize_sphinx(audio_data)
                            print(f"Sphinx Recognition result: {text}")
                        except Exception:
                            text = "Speech recognition failed. Try speaking more clearly."
                    except sr.UnknownValueError:
                        text = "Could not understand audio. Please try speaking more clearly."
                    
                    # Clean up the temporary file
                    try:
                        os.remove(temp_file_path)
                    except Exception as e:
                        print(e)
                        
                    return {"text": text}
            except Exception as sr_error:
                print(f"Error in fallback speech recognition: {sr_error}")

        try:
            os.remove(temp_file_path)
        except Exception as e:
            print(e)
            
        return {"text": "Speech recognition failed. Please try again."}
            
    except Exception as e:
        print(f"Error in audio transcription: {e}")
        import traceback
        traceback.print_exc()
        # Try to clean up temp file in case of error
        try:
            if 'temp_file_path' in locals():
                os.remove(temp_file_path)
        except Exception as e:
            print(e)
        raise HTTPException(status_code=500, detail=str(e))



import spacy

nlp = spacy.load("en_core_web_sm")

def extract_keywords(text, max_keywords=3):
    doc = nlp(text)
    
    keywords = [token.text.lower() for token in doc if token.pos_ in ("NOUN", "PROPN") and not token.is_stop]
    
    return list(dict.fromkeys(keywords))[:max_keywords]  




# Helper functions
def get_embeddings(texts):
    """
    Get embeddings using the local model or fallback to random.
    Always returns a numpy array of shape (len(texts), EMBEDDING_SIZE)
    """
    # Return properly shaped empty array if no texts
    if not texts:
        return np.zeros((0, EMBEDDING_SIZE), dtype=np.float32)
    
    # Create random embeddings to use as fallback
    random_embeddings = np.random.rand(len(texts), EMBEDDING_SIZE).astype(np.float32)
    
    if not USE_LOCAL_EMBEDDINGS:
        print("Using random embeddings (local model disabled)")
        return random_embeddings
    
    try:
        # Process in batches to avoid memory issues with large documents
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            print(f"Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(batch)} texts)")
            
            # Get embeddings for this batch
            batch_embeddings = local_embedding_model.encode(batch, convert_to_numpy=True)
            all_embeddings.append(batch_embeddings)
        
        # Combine all batches
        if all_embeddings:
            return np.vstack(all_embeddings).astype(np.float32)
        else:
            return random_embeddings
            
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        import traceback
        traceback.print_exc()
        return random_embeddings


def summarize_context(context_chain, max_length=500):
    """Create a brief summary of the conversation context."""
    if not context_chain:
        return ""
    
    # Extract exchanges in a readable format
    exchanges = []
    for i in range(0, len(context_chain), 2):
        if i+1 < len(context_chain):
            user_msg = context_chain[i]["content"]
            assistant_msg = context_chain[i+1]["content"]
            exchanges.append(f"User: {user_msg[:50]}... → Assistant: {assistant_msg[:50]}...")
    
    summary = "Previous conversation summary:\n" + "\n".join(exchanges)
    
    # Truncate if too long
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."
    
    return summary


def extract_text_from_file(file_path):
    """Extract text from a file based on its extension."""
    file_extension = os.path.splitext(file_path)[1].lower().replace(".", "")
    
    try:
        if file_extension in ["txt", "md", "csv"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
                
        elif file_extension == "pdf":
            # Try multiple methods for PDF extraction
            text_content = ""
            
            # Method 1: PyPDF2
            try:
                import PyPDF2
                
                with open(file_path, "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    
                    # Extract text from all pages
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
                
                if text_content.strip():
                    print("Successfully extracted text with PyPDF2")
                    return text_content
            except ImportError:
                print("PyPDF2 not installed")
            except Exception as pdf_e:
                print(f"PyPDF2 extraction error: {pdf_e}")
            
            try:
                import pdfplumber
                
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
                
                if text_content.strip():
                    print("Successfully extracted text with pdfplumber")
                    return text_content
            except ImportError:
                print("pdfplumber not installed")
            except Exception as plumber_e:
                print(f"pdfplumber extraction error: {plumber_e}")
            
            # Method 3: Try pdf2image + pytesseract for scanned PDFs
            try:
                from pdf2image import convert_from_path
                import pytesseract
                
                print("Attempting OCR with pdf2image and pytesseract")
                
                images = convert_from_path(file_path)
                for i, image in enumerate(images):
                    page_text = pytesseract.image_to_string(image)
                    if page_text:
                        text_content += page_text + "\n"
                
                if text_content.strip():
                    print("Successfully extracted text with OCR")
                    return text_content
            except ImportError:
                print("pdf2image or pytesseract not installed")
            except Exception as ocr_e:
                print(f"OCR extraction error: {ocr_e}")
            
            if not text_content.strip():
                print("All PDF extraction methods failed, using placeholder text")
                return f"PDF document: {os.path.basename(file_path)} (text extraction failed)"
            
            return text_content
                
        elif file_extension in ["docx", "doc"]:
            # Try to use docx library first
            try:
                import docx
                doc = docx.Document(file_path)
                return "\n\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                print("python-docx not available, using fallback")
            except Exception as docx_e:
                print(f"DOCX extraction error with python-docx: {docx_e}")
            
            # Fallback to Groq
            try:
                with open(file_path, "rb") as f:
                    docx_content = base64.b64encode(f.read()[:5000]).decode('utf-8')
                    
                response = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a document text extraction tool. Extract all text content from the document I provide. Return ONLY the extracted text, nothing else."
                        },
                        {
                            "role": "user",
                            "content": f"This is a document file. Please extract all text content. Here's a base64 preview: {docx_content[:1000]}"
                        }
                    ],
                    model="llama-3.1-8b-instant",
                )
                
                return response.choices[0].message.content
            except Exception as docx_e:
                print(f"DOCX extraction error with Groq: {docx_e}")
                return f"DOCX document: {os.path.basename(file_path)} (text extraction failed)"
        
        # Default: try to read as text
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if content.strip():
                    return content
        except Exception as text_e:
            print(f"Text extraction error: {text_e}")
        
        # If all else fails, return filename as placeholder
        return f"Document: {os.path.basename(file_path)} (content could not be extracted)"
                
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return f"Document: {os.path.basename(file_path)} (extraction error)"


def split_into_chunks(text, chunk_size=1000, overlap=200):  # Increased from 500 to 1000
    """Split text into overlapping chunks while preserving paragraph boundaries."""
    if not text:
        return []
    
    # Split by paragraphs to maintain coherence
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If adding this paragraph exceeds the chunk size and we already have content
        if current_size + len(para) > chunk_size and current_chunk:
            # Save the current chunk
            chunks.append(" ".join(current_chunk))
            
            # Calculate overlap - keep paragraphs until we're under the overlap size
            overlap_count = 0
            overlap_size = 0
            for p in reversed(current_chunk):
                overlap_size += len(p)
                overlap_count += 1
                if overlap_size >= overlap:
                    break
                    
            # Keep the overlap paragraphs
            current_chunk = current_chunk[-overlap_count:] if overlap_count > 0 else []
            current_size = sum(len(p) for p in current_chunk)
        
        # Add the paragraph
        current_chunk.append(para)
        current_size += len(para)
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]
    print(f"Created {len(chunks)} chunks")
    return chunks



def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors, handling shape mismatches."""
    a, b = np.asarray(a), np.asarray(b)
    
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]  # Truncate to match the shortest length
    
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))


def search_documents(query, top_k=5):  # Increased from 3 to 5
    """Search for relevant document chunks based on vector similarity."""
    global document_chunks, document_vectors, document_metadata
    
    # Check if we have documents to search
    if not document_chunks or len(document_vectors) == 0:
        print("No documents to search")
        return []
    
    try:
        # Clean and normalize the query to improve matching
        query = query.strip().lower()
        
        # Get query embedding
        query_vectors = get_embeddings([query])
        
        # Safety check
        if len(query_vectors) == 0:
            print("Failed to get query embeddings")
            return []
            
        query_vector = query_vectors[0]
        
        # Calculate similarities
        similarities = []
        for i, doc_vector in enumerate(document_vectors):
            if i < len(document_chunks):
                similarity = cosine_similarity(query_vector, doc_vector)
                similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in similarities[:top_k]:
            if score > 0:
                # Find which document this chunk belongs to
                for doc_id, meta in document_metadata.items():
                    if idx in meta.get("chunk_indices", []):
                        results.append({
                            "doc_id": doc_id,
                            "filename": meta.get("filename", "Unknown"),
                            "content": document_chunks[idx],
                            "score": float(score)
                        })
                        break
        
        return results
    
    except Exception as e:
        print(f"Error searching documents: {e}")
        import traceback
        traceback.print_exc()
        return []


def save_data():
    """Save vectors and metadata to disk."""
    try:
        # Save document chunks
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(document_chunks, f, ensure_ascii=False)
        
        # Save metadata
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(document_metadata, f, ensure_ascii=False)
        
        # Save vectors - ensure it's not empty
        if document_vectors.size > 0:
            np.savez_compressed(VECTORS_FILE, vectors=document_vectors)
            
        print("Data saved successfully")
    except Exception as e:
        print(f"Error saving data: {e}")
        import traceback
        traceback.print_exc()


def process_document(file_path, file_name, doc_id=None, file_info=None):
    """Process a document and add it to the vector store."""
    global document_chunks, document_vectors, document_metadata
    
    try:
        print(f"Processing document: {file_name}")
        
        # Extract text content
        text_content = extract_text_from_file(str(file_path))
        
        # Lower the minimum text requirement from 50 to 5 characters
        if not text_content or len(text_content) < 5:
            print(f"Error: Not enough content extracted from {file_name}")
            return False
        
        # Split into chunks - even if it's just one small chunk
        chunks = split_into_chunks(text_content, chunk_size=500)  # Smaller chunk size
        
        # If no chunks were created, force at least one chunk with the content
        if not chunks:
            chunks = [text_content]
            print(f"Created 1 chunk manually from document")
        
        # Generate a document ID if not provided
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        # Get vectors for new chunks
        new_vectors = get_embeddings(chunks)
        
        # Verify vectors shape
        if new_vectors.shape[0] != len(chunks):
            print(f"Warning: Vector count mismatch. Got {new_vectors.shape[0]} vectors for {len(chunks)} chunks.")
            # Fix by creating random vectors of correct size
            new_vectors = np.random.rand(len(chunks), EMBEDDING_SIZE).astype(np.float32)
        
        # Calculate chunk indices
        start_idx = len(document_chunks)
        chunk_indices = list(range(start_idx, start_idx + len(chunks)))
        
        # Add document metadata
        document_metadata[doc_id] = {
            "filename": file_name,
            "upload_date": datetime.now().isoformat(),
            "chunk_indices": chunk_indices
        }
        
        # Add additional file info if provided
        if file_info:
            document_metadata[doc_id].update(file_info)
        
        # Add chunks to global list
        document_chunks.extend(chunks)
        
        # Update vectors safely
        try:
            if document_vectors.size == 0:
                # First document - just use the new vectors
                document_vectors = new_vectors
            else:
                # Check dimensions before stacking
                if document_vectors.shape[1] != new_vectors.shape[1]:
                    print(f"Dimension mismatch: {document_vectors.shape} vs {new_vectors.shape}")
                    # Reshape new vectors to match existing dimension
                    reshaped_vectors = np.random.rand(len(chunks), document_vectors.shape[1]).astype(np.float32)
                    document_vectors = np.vstack([document_vectors, reshaped_vectors])
                else:
                    # Normal case - dimensions match
                    document_vectors = np.vstack([document_vectors, new_vectors])
        except Exception as vector_error:
            print(f"Error updating vectors: {vector_error}")
            # Fallback: recreate vectors for all documents
            document_vectors = np.random.rand(len(document_chunks), EMBEDDING_SIZE).astype(np.float32)
        
        # Save data
        save_data()
        
        print(f"Successfully processed document: {file_name} into {len(chunks)} chunks")
        return True
    
    except Exception as e:
        print(f"Error processing document {file_name}: {e}")
        import traceback
        traceback.print_exc()
        return False



@app.post("/upload_document")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document for RAG."""
    print(f"Received upload request for file: {file.filename}")
    
    # Generate a document ID first
    doc_id = str(uuid.uuid4())
    
    # Generate a more meaningful filename that includes the document ID
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    safe_filename = file.filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
    unique_filename = f"{timestamp}_{doc_id[:8]}_{safe_filename}"  # Include first 8 chars of ID
    file_path = UPLOAD_DIR / unique_filename
    
    # Save uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"Saved file as: {unique_filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # Before processing, store the file information including path
    # This will be stored in metadata after successful processing
    file_info = {
        "filename": file.filename,
        "stored_filename": unique_filename,
        "upload_date": datetime.now().isoformat(),
        "file_path": str(file_path),
        "doc_type": os.path.splitext(file.filename)[1].lower().replace(".", "") if "." in file.filename else "unknown"
    }
    
    # Process document and add to vector store
    success = process_document(file_path, file.filename, doc_id, file_info)
    
    if not success:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail="Failed to process document")
    
    # Return the document info
    return {
        "id": doc_id,
        "filename": file.filename,
        "upload_date": file_info["upload_date"],
        "doc_type": file_info["doc_type"]
    }


@app.get("/documents", response_model=List[DocumentResponse])
async def list_documents():
    """List all uploaded documents."""
    documents = []
    for doc_id, meta in document_metadata.items():
        documents.append({
            "id": doc_id,
            "filename": meta.get("filename", "Unknown"),
            "upload_date": meta.get("upload_date", datetime.now().isoformat()),
            "doc_type": os.path.splitext(meta.get("filename", ""))[1].lower().replace(".", "") if "." in meta.get("filename", "") else "unknown"
        })
    return documents


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document by ID."""
    global document_chunks, document_vectors, document_metadata
    
    if document_id not in document_metadata:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Get document metadata
        doc_meta = document_metadata[document_id]
        chunk_indices = sorted(doc_meta.get("chunk_indices", []))
        print(f"Deleting document {document_id} with {len(chunk_indices)} chunks")
        
        # Try to find and delete the physical file
        file_deleted = False
        
        # Method 1: Use stored file path if available
        if "file_path" in doc_meta:
            try:
                file_path = Path(doc_meta["file_path"])
                if file_path.exists():
                    os.remove(file_path)
                    file_deleted = True
                    print(f"Deleted file using stored path: {file_path}")
            except Exception as path_error:
                print(f"Error deleting file using stored path: {path_error}")
        
        # Method 2: Use stored filename if available
        if not file_deleted and "stored_filename" in doc_meta:
            try:
                file_path = UPLOAD_DIR / doc_meta["stored_filename"]
                if file_path.exists():
                    os.remove(file_path)
                    file_deleted = True
                    print(f"Deleted file using stored filename: {file_path}")
            except Exception as name_error:
                print(f"Error deleting file using stored filename: {name_error}")
        
        # Method 3: Look for document ID in filenames (fallback)
        if not file_deleted:
            for file_path in UPLOAD_DIR.iterdir():
                if document_id in str(file_path):
                    try:
                        os.remove(file_path)
                        file_deleted = True
                        print(f"Deleted file by searching for ID: {file_path}")
                    except Exception as del_error:
                        print(f"Error removing file: {del_error}")
                    break
        
        # Remove metadata
        del document_metadata[document_id]
        
        if not chunk_indices:
            # No chunks to remove
            print("No chunks to remove, only deleted metadata")
            save_data()
            return {
                "status": "success", 
                "message": "Document metadata deleted",
                "file_deleted": file_deleted
            }
        
        # Create sets for faster lookups
        chunk_indices_set = set(chunk_indices)
        
        # Remove chunks and adjust indices in other documents
        new_chunks = []
        new_vectors_list = []
        index_map = {}  # Maps old indices to new indices
        
        # Build new chunks and vectors arrays without the deleted chunks
        for old_idx in range(len(document_chunks)):
            if old_idx not in chunk_indices_set:
                new_idx = len(new_chunks)
                index_map[old_idx] = new_idx
                new_chunks.append(document_chunks[old_idx])
                # Check if we have vectors and if the index is valid
                if document_vectors is not None and document_vectors.size > 0 and old_idx < len(document_vectors):
                    new_vectors_list.append(document_vectors[old_idx])
        
        # Update chunk indices in metadata
        updated_metadata = {}
        for doc_id, meta in document_metadata.items():
            old_indices = meta.get("chunk_indices", [])
            new_indices = []
            
            for idx in old_indices:
                if idx in index_map:
                    new_indices.append(index_map[idx])
            
            if new_indices:  # Only keep documents that still have chunks
                meta["chunk_indices"] = new_indices
                updated_metadata[doc_id] = meta
            else:
                print(f"Warning: Document {doc_id} has no chunks after deletion")
        
        # Update global variables
        document_metadata = updated_metadata
        document_chunks = new_chunks
        
        # Update vectors with proper error handling
        try:
            if new_vectors_list:
                document_vectors = np.array(new_vectors_list, dtype=np.float32)
                print(f"Updated vectors, new shape: {document_vectors.shape}")
            else:
                document_vectors = np.zeros((0, EMBEDDING_SIZE), dtype=np.float32)
                print("No vectors remaining, initialized empty array")
        except Exception as vec_error:
            print(f"Error updating vectors after deletion: {vec_error}")
            # Create new random vectors as fallback
            document_vectors = np.zeros((len(new_chunks), EMBEDDING_SIZE), dtype=np.float32)
            print(f"Created fallback vectors with shape {document_vectors.shape}")
        
        # Save updated data
        save_data()
        
        return {
            "status": "success", 
            "message": "Document deleted successfully",
            "chunks_removed": len(chunk_indices),
            "file_deleted": file_deleted
        }
    
    except Exception as e:
        print(f"Error deleting document: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")
    
    except Exception as e:
        print(f"Error deleting document: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@app.post("/chat")
async def chat_with_context_filtering(request: ChatRequest):
    """Enhanced chat endpoint that uses conversation context to improve document search."""
    try:
        # Initialize context chain if not provided
        context_chain = request.context_chain or []
        
        # Extract context to enhance the search query
        enhanced_query = request.message
        
        # If we have context, extract recent topics to improve search
        if context_chain:
            # Get the last few user messages to extract topics
            recent_user_messages = [
                msg["content"] for msg in context_chain[-6:] 
                if msg["role"] == "user"
            ]
            
            # Combine with current query to create an enhanced search query
            if recent_user_messages:
                # Extract key terms from recent messages
                recent_topics = extract_keywords(" ".join(recent_user_messages), max_keywords=3)
                if recent_topics:
                    # Add recent topics to the search query
                    enhanced_query = f"{request.message} {' '.join(recent_topics)}"
        
        # Search for relevant document chunks with the enhanced query
        search_results = search_documents(enhanced_query, top_k=5)
        
        # Continue with the regular processing as in the chat endpoint...
        # (rest of the function is identical to the chat endpoint above)
        # ...
        
        # Format document context
        if search_results:
            context_parts = []
            for i, result in enumerate(search_results, 1):
                context_parts.append(
                    f"Document {i}: {result['filename']}\n"
                    f"{result['content']}"
                )
            
            doc_context = "\n\n---\n\n".join(context_parts)
            
            # Build the system prompt with documents and chain instructions
            system_prompt = f"""You are a helpful assistant with access to the following document excerpts:

---BEGIN DOCUMENTS---
{doc_context}
---END DOCUMENTS---

INSTRUCTIONS:
1. Answer the user's question primarily using information from these documents.
2. If the documents contain the information, cite the specific document by saying "According to Document X" in your response.
3. If the documents don't fully answer the question, clearly state what information is missing.
4. If the documents don't provide any relevant information, inform the user and provide a general response based on your knowledge.
5. Be concise but thorough, focusing on the most relevant information to answer the question.
6. Consider the conversation context when answering. This might be a follow-up question.
"""
        else:
            # No relevant documents found
            system_prompt = """You are a helpful assistant. The user has asked a question, but I don't have any relevant documents in my knowledge base to address this specific query. I'll provide a helpful response based on my general knowledge. Consider the conversation context when answering. This might be a follow-up question."""
        
        # Build the LLM messages array
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add context chain messages to preserve conversation flow
        # Limit to the specified context window size
        for message in context_chain[-request.context_window:]:
            messages.append(message)
        
        # Add the current user message
        messages.append({"role": "user", "content": request.message})
        
        # Call Groq API for completion
        response = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-8b-instant",
            max_tokens=1024,
            temperature=0.7,
        )
        
        assistant_response = response.choices[0].message.content
        
        # Update the context chain with the new exchange
        updated_context_chain = context_chain.copy()
        updated_context_chain.append({"role": "user", "content": request.message})
        updated_context_chain.append({"role": "assistant", "content": assistant_response})
        
        # Return both the response and the updated context chain
        return {
            "response": assistant_response,
            "context_chain": updated_context_chain  # Return this so the client can include it in the next request
        }
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    vector_shape = None if document_vectors.size == 0 else document_vectors.shape
    
    return {
        "status": "ok", 
        "documents": len(document_metadata),
        "chunks": len(document_chunks),
        "vectors_shape": str(vector_shape)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)