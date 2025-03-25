import glob, os
import PyPDF2

class PDFReader:
    def readfiles(self, path):
        pdfs = []
        for file in glob.glob(os.path.join(path, "*.pdf")):
            pdfs.append(file)
        return pdfs
    
    def openFile(self, file_path):
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = text + "\n" + page.extract_text()
        return text