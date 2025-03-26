import glob, os
import PyPDF2
import json

class PDFReader:
    """
        Reads all pdf file names from directory and load files as a text
    """
    
    def readfiles(self, path, extension):
        pdfs = []
        for file in glob.glob(os.path.join(path, extension)):
            pdfs.append(file)
        return pdfs
    
    def openPDFFile(self, file_path):
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = text + "\n" + page.extract_text()
        return text

    def openTXTFile(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def openJSONFile(self, file_path):
        """Opens and reads a JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # Load JSON data
            return json.dumps(data)  # Convert JSON to string