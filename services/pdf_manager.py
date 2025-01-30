import os
from PyPDF2 import PdfReader
class PDFManger:
    def __init__(self, docs_dir: str = "documents/"):
        self.docs_dir = docs_dir

    def list_all_docs(self):
        """
        Lists all PDF documents in the directory
        """
        docs = []
        for doc in os.listdir(self.docs_dir):
            if doc.endswith('.pdf'):
                docs.append(doc)
        return docs
    
    def read_pdf(self, filename):
        """
        Reads and returns the text content from PDF file
        """
        pdf_path = os.path.join(self.docs_dir, filename)
        # check if file exists or not
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File {filename} not found in {self.docs_dir}")
        # read all the pages of the PDF file
        reader = PdfReader(pdf_path)
        content = "\n".join([page.extract_text() for page in reader.pages])
        return content