import logging
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from google.cloud import documentai_v1beta3 as documentai
from qdrant_client import QdrantClient
from settings import env_settings

logger = logging.getLogger(__name__)

QDRANT_URL = env_settings.QDRANT_URL
QDRANT_API_KEY = env_settings.QDRANT_API_KEY
OPENAI_API_KEY = env_settings.OPENAI_API_KEY
GCP_PROJECT_ID = env_settings.GCP_PROJECT_ID
GCP_LOCATION = env_settings.GCP_LOCATION
GCP_PROCESSOR_ID = env_settings.GCP_PROCESSOR_ID
GCP_PROCESSOR_VERSION = env_settings.GCP_PROCESSOR_VERSION


class EthanRAG:
    _qdrant_client: QdrantClient = None

    def __init__(self):
        if EthanRAG._qdrant_client is None:
            EthanRAG._qdrant_client = self.connect_qdrant()

    @staticmethod
    def connect_qdrant() -> QdrantClient:
        return QdrantClient(
            api_key=QDRANT_API_KEY,
            url=QDRANT_URL,
        )

    @staticmethod
    async def extract_text_from_pdf(file_path: str) -> str:
        try:
            logger.info(f"Processing with Google Document AI: {file_path}")
            client = documentai.DocumentProcessorServiceClient()
            name = client.processor_version_path(
                GCP_PROJECT_ID, GCP_LOCATION, GCP_PROCESSOR_ID, GCP_PROCESSOR_VERSION
            )
            with open(file_path, "rb") as pdf_file:
                raw_document = documentai.RawDocument(
                    content=pdf_file.read(), mime_type="application/pdf"
                )
            request = documentai.ProcessRequest(
                name=name,
                raw_document=raw_document,
                process_options=documentai.ProcessOptions(
                    ocr_config=documentai.OcrConfig(
                        enable_native_pdf_parsing=True)
                ),
            )
            result = client.process_document(request=request)
            document = result.document
            extracted_text = document.text
            return extracted_text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise e

    async def process_and_store_pdf(file_path: str, filename: str):
        try:
            extracted_text = await EthanRAG.extract_text_from_pdf(file_path)
            text_splitter = CharacterTextSplitter(
                chunk_size=20,
                chunk_overlap=0,
                strip_whitespace=False,
                separator=""
            )
            chunks = text_splitter.create_documents([extracted_text])
            embeddings = OpenAIEmbeddings(
                api_key=OPENAI_API_KEY, 
                model="text-embedding-3-large"
            )
            qdrant_client = EthanRAG._qdrant_client
            qdrant_client.create_collection(
                collection_name=filename,
                embedding_function=embeddings,
                vector_size=1536
            )
            qdrant_client.upload_documents(
                collection_name=filename,
                documents=chunks,
                embeddings=embeddings,
                metadata=[{"filename": filename}] * len(chunks)
            )
            logger.info(f"Successfully processed and stored PDF: {filename}")

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise e
