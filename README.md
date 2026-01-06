# Project Manise

A Python pipeline for extracting, parsing, cleaning, and processing dictionary data from PDF dictionaries of Ambonese Malay (a dialect of Indonesian). The pipeline converts raw dictionary text into structured data and generates parallel sentence pairs for language processing tasks.

## Overview

This project processes dictionary PDFs through a 5-step pipeline:

1. **Text Extraction** - Extracts text from PDF dictionary (handles double-column layout)
2. **Entry Parsing** - Uses Google Gemini API to parse raw text into structured JSON entries
3. **Placeholder Cleaning** - Cleans OCR artifacts and standardizes placeholder symbols
4. **OCR/Typo Correction** - Corrects OCR errors while preserving dialectal authenticity
5. **Parallel Sentence Extraction** - Extracts Ambonese-Indonesian sentence pairs for corpus creation

## Features

- **LLM-Assisted Processing**: Uses Google Gemini API for intelligent parsing and correction
- **Dialect Preservation**: Maintains authentic Ambonese Malay dialectal variations while fixing OCR errors
- **Robust Error Handling**: Progress tracking, resume capability, and comprehensive logging
- **Multiple Output Formats**: Supports JSON, JSONL, and CSV formats
- **Parallel Corpus Generation**: Creates aligned sentence pairs for NLP applications

## Requirements

- Python 3.7+
- Google Gemini API key (set as `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variable)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/project-manise.git
cd project-manise
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API key:
```bash
export GEMINI_API_KEY="your-api-key-here"
# or
export GOOGLE_API_KEY="your-api-key-here"
```

## Usage

### Step 1: Extract Dictionary Text

Extract text from the PDF dictionary starting at a specified page:

```bash
python 1_extract_dictionary_text.py dictionary_20260105.pdf --start-page 16
```

**Output**: `outputs/extractions/extraction_TIMESTAMP.txt`

### Step 2: Parse Dictionary Entries

Parse the extracted text into structured JSON entries:

```bash
python 2_parse_dictionary_entries.py
```

This script automatically finds the most recent extraction file. It uses Google Gemini API to parse entries with:
- Headwords
- Definitions
- Ambonese Malay example sentences
- Indonesian translations
- Sense numbers

**Output**: `outputs/parsed/entries_TIMESTAMP.json`

### Step 3: Clean Placeholders

Clean OCR artifacts and standardize placeholder symbols:

```bash
python 3_clean_placeholders.py outputs/parsed/entries_TIMESTAMP.json
```

**Output**: `outputs/cleaned/progress_TIMESTAMP_original_cleaned.json`

### Step 3.1: Replace Dash Patterns (Optional)

Replace " - " patterns with headwords and "/" with "l":

```bash
python 3.1_replace_dash_space_dash.py outputs/cleaned/progress_TIMESTAMP_original_cleaned.json
```

**Output**: Updated cleaned JSON file

### Step 4: Correct OCR Errors and Typos

Correct OCR errors while preserving dialectal variations:

```bash
python 4_correct_ocr_typos.py outputs/cleaned/progress_TIMESTAMP_original_cleaned.json
```

**Options**:
- `--batch-size N`: Number of examples per API call (default: 15)
- `--model MODEL`: Gemini model name (default: gemini-2.0-flash-exp)
- `--output-dir DIR`: Output directory (default: outputs/corrected)

**Output**: 
- `outputs/corrected/corrections_TIMESTAMP.jsonl` (incremental)
- `outputs/corrected/corrections_TIMESTAMP.json` (consolidated)

**Note**: This script preserves authentic Ambonese dialectal variations (e.g., "kalo", "pigi", "sa") while standardizing Indonesian translations.

### Step 5: Extract Parallel Sentences

Extract parallel Ambonese-Indonesian sentence pairs:

```bash
python 5_extract_parallel_sentences.py outputs/cleaned/progress_TIMESTAMP_original_cleaned.json --count 100
```

**Options**:
- `--count N`: Number of sentence pairs to extract (default: 100)
- `--output FILE`: Output file path (default: outputs/parallel_sentences_100.jsonl)

**Output**: `outputs/parallel_sentences_100.jsonl`

Each line contains a JSON object:
```json
{"ambonese": "Beta pigi ka pasar", "indonesian": "Saya pergi ke pasar"}
```

## Utility Scripts

### Convert JSON to CSV

Convert dictionary JSON to CSV format (flattens meanings):

```bash
python convert_json_csv.py json_to_csv input.json output.csv
```

Convert CSV back to JSON:

```bash
python convert_json_csv.py csv_to_json input.csv output.json
```

Convert CSV to JSONL:

```bash
python convert_json_csv.py csv_to_jsonl input.csv output.jsonl
```

## Project Structure

```
project-manise/
├── 1_extract_dictionary_text.py      # Step 1: PDF text extraction
├── 2_parse_dictionary_entries.py     # Step 2: Entry parsing with LLM
├── 3_clean_placeholders.py            # Step 3: Clean OCR artifacts
├── 3.1_replace_dash_space_dash.py    # Step 3.1: Replace dash patterns
├── 4_correct_ocr_typos.py             # Step 4: OCR/typo correction
├── 5_extract_parallel_sentences.py    # Step 5: Extract parallel sentences
├── convert_json_csv.py                # Utility: JSON/CSV conversion
├── utils/                             # Utility modules
│   ├── change_tracker.py
│   ├── file_manager.py
│   ├── logger.py
│   └── validators.py
├── visualization/                    # Visualization scripts
│   └── test_embeddings_google.py
├── outputs/                           # Generated outputs (excluded from git)
│   └── parallel_sentences_100.jsonl   # Sample output (included)
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Data Format

### Dictionary Entry JSON Structure

```json
{
  "entries": [
    {
      "headword": "pigi",
      "page_number": 123,
      "meanings": [
        {
          "sense_number": 1,
          "definition": "pergi",
          "ambonese_example": "Beta pigi ka pasar",
          "indonesian_translation": "Saya pergi ke pasar"
        }
      ]
    }
  ]
}
```

### Parallel Sentences JSONL Format

Each line is a JSON object:
```json
{"ambonese": "Beta pigi ka pasar", "indonesian": "Saya pergi ke pasar"}
```

## Notes

- **Dialect Preservation**: The correction script (Step 4) is designed to preserve authentic Ambonese Malay dialectal variations. It only fixes obvious OCR errors, not dialectal spelling differences.
- **Resume Capability**: Scripts support resuming from interruptions by tracking progress in output files.
- **Logging**: All scripts generate detailed logs in `outputs/logs/` with timestamps.

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

