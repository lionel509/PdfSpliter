import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from PyPDF2 import PdfReader  # Importing PDF processing library

# Ensure required NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self, lower_case=True, remove_stopwords=True):
        """
        Initializes the text preprocessor.
        :param lower_case: Whether to convert text to lowercase.
        :param remove_stopwords: Whether to remove stopwords.
        """
        self.lower_case = lower_case
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words('english'))

    def process_text(self, text):
        """
        Processes a single text input.
        :param text: The input text string.
        :return: List of processed tokens.
        """
        # Convert text to lowercase
        if self.lower_case:
            text = text.lower()

        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stopwords
        if self.remove_stopwords:
            tokens = [word for word in tokens if word not in self.stop_words]

        return tokens

    def process_pdf(self, pdf_path):
        """
        Processes a PDF file to extract text content.
        :param pdf_path: Path to the PDF file.
        :return: Extracted text as a single string.
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return self.process_text(text)
        except Exception as e:
            raise ValueError(f"Error processing PDF: {e}")
