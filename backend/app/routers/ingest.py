from fastapi import APIRouter, UploadFile, File, HTTPException
import boto3
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.ml.embeddings import HuggingFaceEmbeddingClient
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional
from app.config import Config

router = APIRouter()
logger = logging.getLogger(__name__)

# Lazy initialization
s3_client = None
embedding_model = None

# Registry of ingested documents in this session
ingested_documents_registry: List[Dict] = []

def get_s3_client():
    global s3_client
    if s3_client is None:
        if Config.AWS_ACCESS_KEY_ID and Config.AWS_SECRET_ACCESS_KEY:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
                region_name=Config.AWS_REGION
            )
        else:
            return None
    return s3_client

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddingClient(
            token=Config.HF_TOKEN,
            model_name="BAAI/bge-large-en-v1.5"
        )
    return embedding_model


@router.post("/upload-pdf")
@router.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    Ingest a PDF or document:
    1. Optionally upload raw file to AWS S3 backup (if AWS keys configured)
    2. Extract text from PDF / text document
    3. Split text into overlapping semantic chunks
    4. Generate 1024-dim embeddings using Hugging Face Inference API (BAAI/bge-large-en-v1.5)
    5. Upsert chunks with metadata into Pinecone vector database
    """
    valid_extensions = ('.pdf', '.txt', '.md', '.csv', '.json', '.doc', '.docx')
    if not file.filename.lower().endswith(valid_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Please upload {', '.join(valid_extensions)}"
        )
        
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        file_key = f"documents/{uuid.uuid4().hex[:8]}-{file.filename}"
        s3_uploaded = False
        
        # 1. Upload to S3 Backup if credentials exist
        if Config.AWS_ACCESS_KEY_ID and Config.AWS_SECRET_ACCESS_KEY:
            try:
                s3 = get_s3_client()
                if s3:
                    content_type = file.content_type or "application/octet-stream"
                    s3.put_object(
                        Bucket=Config.S3_BUCKET_NAME,
                        Key=file_key,
                        Body=content,
                        ContentType=content_type
                    )
                    s3_uploaded = True
                    logger.info(f"✅ Uploaded to S3: s3://{Config.S3_BUCKET_NAME}/{file_key}")
            except Exception as s3_err:
                logger.warning(f"⚠️ S3 backup skipped due to error: {s3_err}")
        else:
            logger.info("ℹ️ AWS credentials not configured. Skipping S3 backup.")
        
        # 2. Extract Text
        filename_lower = file.filename.lower()
        text = ""
        
        if filename_lower.endswith('.pdf'):
            try:
                doc = fitz.open(stream=content, filetype="pdf")
                for page in doc:
                    text += page.get_text() + "\n"
            except Exception as pdf_err:
                logger.error(f"Failed to parse PDF: {pdf_err}")
                raise HTTPException(status_code=400, detail=f"Could not parse PDF content: {pdf_err}")
        else:
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1', errors='ignore')

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No extractable text was found in the uploaded file (may be scanned images)."
            )

        # 3. Semantic Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        total_chunks = len(chunks)
        
        if total_chunks == 0:
            raise HTTPException(status_code=400, detail="Document text could not be chunked.")

        # 4. Embed with Hugging Face & Upsert to Pinecone
        from app.main import persistence
        if not persistence.index:
            raise HTTPException(
                status_code=500,
                detail="Pinecone vector database is not connected. Check PINECONE_API_KEY in .env"
            )
            
        model = get_embedding_model()
        batch_size = 20
        ingested_count = 0
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            embeddings = model.embed_documents(batch)
            
            vectors = []
            for j, emb in enumerate(embeddings):
                chunk_index = i + j
                vector_id = f"doc_{uuid.uuid4().hex[:8]}_chunk_{chunk_index}"
                vectors.append({
                    "id": vector_id,
                    "values": emb,
                    "metadata": {
                        "source": file.filename,
                        "text": batch[j],
                        "type": "document_chunk",
                        "chunk_index": chunk_index,
                        "total_chunks": total_chunks,
                        "s3_key": file_key if s3_uploaded else "",
                        "timestamp": datetime.now().isoformat()
                    }
                })
            
            persistence.index.upsert(vectors=vectors)
            ingested_count += len(vectors)
            
        doc_entry = {
            "id": str(uuid.uuid4()),
            "filename": file.filename,
            "chunks_count": total_chunks,
            "characters_count": len(text),
            "size_bytes": len(content),
            "s3_uploaded": s3_uploaded,
            "s3_key": file_key if s3_uploaded else None,
            "timestamp": datetime.now().isoformat()
        }
        ingested_documents_registry.insert(0, doc_entry)
        
        logger.info(f"🎉 Successfully ingested '{file.filename}' ({total_chunks} chunks) into Pinecone")
        
        return {
            "status": "success",
            "message": f"Successfully ingested {total_chunks} chunks into Pinecone vector database" + (" and S3" if s3_uploaded else ""),
            "document": doc_entry
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def get_ingested_documents():
    """List of documents ingested in the current session"""
    return {"documents": ingested_documents_registry}


@router.get("/search")
async def search_documents(query: str, top_k: int = 4):
    """Semantic similarity search endpoint for uploaded documents"""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    from app.main import persistence
    results = persistence.similarity_search(query=query, top_k=top_k)
    return {"query": query, "count": len(results), "results": results}
