from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader

def load_all_documents(data_dir:str)->List[Any]:
    data_path = Path(data_dir).resolve()
    data_pdf = list(data_path.glob('**/*.pdf'))
    documents = []

    for pdf_files in data_pdf:
        print(f"Loading Pdf:{pdf_files}")
        try:
            loaders = PyPDFLoader(str(pdf_files)).load()
            documents.extend(loaders)
        except Exception as e:
            print(f'failed to Load Pdf:{pdf_files}:{e}')

    return documents

