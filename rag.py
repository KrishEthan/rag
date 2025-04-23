import logging
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from google.cloud import documentai_v1beta3 as documentai
from qdrant_client import QdrantClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
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

    def get_documents(self, extracted_text: str) -> list[Document]:
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            documents = text_splitter.split_text(extracted_text)
            return [Document(page_content=text) for text in documents]
        except Exception as e:
            logger.error(f"Error splitting text into documents: {e}")
            raise e
        
    def create_qdrant_index(self, documents: list[Document], filename: str):
        try:
            logger.info(f"Creating Qdrant index for {filename}")
            qdrant = Qdrant.from_documents(
                documents,
                api_key=QDRANT_API_KEY,
                url=QDRANT_URL,
                embedding=OpenAIEmbeddings(
                    openai_api_key=OPENAI_API_KEY,
                ),
                collection_name="ethan-rag",
            )
            logger.info(f"Qdrant index created for {filename}")
            return qdrant
        except Exception as e:
            logger.error(f"Error creating Qdrant index: {e}")
            raise e

    def extract_text(self, file_path: str) -> str:
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
        
    def process_and_store_pdf(self, file_path: str, filename: str):
        try:
            logger.info(f"Processing and storing PDF: {file_path}")
            extracted_text = self.extract_text(file_path)
            documents = self.get_documents(extracted_text)
            self.create_qdrant_index(documents, filename)
            logger.info(f"PDF processed and stored successfully: {filename}")
        except Exception as e:
            logger.error(f"Error processing and storing PDF: {e}")
            raise e
        
    def query(self, query: str, k: int = 5):
        try:
            logger.info(f"Querying Qdrant index with query: {query}")
            qdrant = Qdrant(
                client=QdrantClient(
                    api_key=QDRANT_API_KEY,
                    url=QDRANT_URL,
                ),
                embeddings=OpenAIEmbeddings(
                    openai_api_key=OPENAI_API_KEY,
                ),
                collection_name="ethan-rag",
            )
            results = qdrant.similarity_search(query, k=k)
            return results
        except Exception as e:
            logger.error(f"Error querying Qdrant index: {e}")
            raise e
    
