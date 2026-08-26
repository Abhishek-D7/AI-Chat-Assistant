from fastapi import APIRouter, UploadFile, File, HTTPException
import boto3
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import uuid
import logging
from app.config import Config

router = APIRouter()
logger = logging.getLogger(__name__)

# Lazy initialization
s3_client = None
embedding_model = None

def get_s3_client():
    global s3_client
    if s3_client is None:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            region_name=Config.AWS_REGION
        )
    return s3_client

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model

@router.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        # Read file
        content = await file.read()
        
        # 1. Upload to S3 Backup
        s3 = get_s3_client()
        file_key = f"pdfs/{uuid.uuid4()}-{file.filename}"
        s3.put_object(
            Bucket=Config.S3_BUCKET_NAME,
            Key=file_key,
            Body=content,
            ContentType="application/pdf"
        )
        logger.info(f"✅ Uploaded to S3: {file_key}")
        
        # 2. Extract Text with PyMuPDF
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
            
        # 3. Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # 4. Embed & Store in Pinecone
        from app.main import persistence
        if not persistence.index:
            raise HTTPException(status_code=500, detail="Pinecone is not initialized")
            
        model = get_embedding_model()
        batch_size = 50
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i+batch_size]
            embeddings = model.encode(batch).tolist()
            
            vectors = []
            for j, emb in enumerate(embeddings):
                vector_id = f"{file_key}-chunk-{i+j}"
                vectors.append({
                    "id": vector_id,
                    "values": emb,
                    "metadata": {
                        "source": file.filename,
                        "text": batch[j],
                        "type": "document_chunk"
                    }
                })
            
            persistence.index.upsert(vectors=vectors)
            
        return {"status": "success", "message": f"Ingested {total_chunks} chunks", "s3_key": file_key}
        
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
