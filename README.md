# PdfSpliter

PdfSpliter is a Python application designed to extract and process questions from PDF files. It supports recognizing multiple-choice questions, extracting answer tables, and exporting structured data in formats like JSON and CSV. Additionally, it includes a web interface for uploading PDFs and viewing results.

---

## Features

- **PDF Parsing**: Extracts text and images from PDFs for processing.
- **Question Recognition**: Detects multiple-choice questions and organizes them into structured formats.
- **Answer Table Extraction**: Extracts answers and maps them to corresponding questions.
- **Export Formats**: Supports output in JSON and CSV for easy integration with other tools.
- **Web Interface**: User-friendly interface for uploading PDFs and viewing results in real-time.

## Installation

### Prerequisites

Ensure you have the following installed on your system:

- Python 3.8 or later
- pip (Python package manager)

### Steps
   ```sh
   git clone https://github.com/lionel509/PdfSpliter.git
   ```

1. Install dependencies:

```sh
   pip install -r requirements.txt
   ```

2. Run the application:\

```sh
   python app.py


## Usage

### Command-Line Interface

You can use PdfSpliter via the command line:
You can use PdfSpliter via the command line:

```sh
python main.py --input <path-to-pdf> --output <output-format>
#### Options



- `--input`: Specify the path to the input PDF file.
- '--output': Specify the output format 
#### Example


```sh
python main.py --input example.pdf --output json
```

### Web Interface

1. Start the web application:
   ```sh
   python app.py
   ```

2. Open your browser and navigate to:
   [http://localhost:5000](http://localhost:5000)

3. Upload a PDF file and view the extracted data directly in the interface.

### Input PDF

Upload a PDF with the following content:

### Output JSON

```json
{
  "questions": [
    {
      "question": "What is the capital of France?",
      "options": ["A) Berlin", "B) Paris", "C) Rome", "D) Madrid"],
      "answer": "B"
    }
  ]
### Output CSV
| Question                      | Option A | Option B | Option C | Option D | Answer |

### Output CSV
| Question                      | Option A | Option B | Option C | Option D | Answer |
|-------------------------------|----------|----------|----------|----------|--------|
| What is the capital of France? | Berlin   | Paris    | Rome     | Madrid   | B      |

## Contributing

Contributions are welcome! Follow these steps to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```sh
   git checkout -b feature-name
   ```

1. Commit your changes:

   ```sh
   git commit -m "Add feature-name"
   ```

2. Push to your branch:

   ```sh
   git push origin feature-name
   ```

3. Open a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions, issues, or feedback, feel free to reach out:

- **Email**: [lionel509@example.com](mailto:lionel509@example.com)
- **GitHub Issues**: Submit an Issue
