# Project Manise 🏝️

Turning static PDFs into a dynamic parallel corpus for Ambonese Malay.

Project Manise is a Python pipeline designed to rescue dictionary data from the clutches of PDF formatting. It extracts, parses, cleans, and aligns text from Ambonese Malay dictionaries, converting them into high-quality, structured JSON and parallel sentence pairs.

## The Mission:
The primary purpose of this project is to collect and structure authentic linguistic data. By converting static documentation into machine-readable formats, we create a foundational dataset that can be augmented to train more robust language models on under-represented languages.

"Manise" implies sweetness and beauty—we're taking messy, raw data and making it sweet, structured, and ready for training.

## 🧐 What is this?

Ambonese Malay is a vibrant language, but like many regional tongues, it is under-resourced in the digital space. This project bridges the gap by:

- Ingesting raw dictionary PDFs.
- Structuring raw text into semantic components (Headwords, Definitions, Examples).
- Generating clean parallel corpora (Ambonese <-> Indonesian) for NLP tasks.

## 🚀 The Pipeline

We process data moving from raw extraction to structured gold:

- **📄 Text Extraction**: Rips text from double-column PDFs (even the messy parts).
- **🧠 Semantic Parsing**: Identifies linguistic categories (Headwords vs. Definitions) and structures the text into JSON.
- **🧹 The Cleanup**: Hunts down OCR artifacts. Turns placeholder symbols into actual words and fixes spacing issues.
- **📝 Correction**: Fixes OCR errors and creates a clean text stream.
- **✨ Corpus Generation**: Extracts aligned sentence pairs (Ambonese -> Indonesian) for immediate use in NLP training.

## 🛠️ Requirements

- Python 3.7+
- LLM API Key (Set as `GEMINI_API_KEY` or `GOOGLE_API_KEY`).

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/dansachs/project-manise.git
cd project-manise

# Install the dependencies
pip install -r requirements.txt

# Set your API Key
export GEMINI_API_KEY="your-api-key-here"
```

## 💻 Usage Guide

### 1. Extract (Get the text out)

Pull raw text from the PDF, handling columns automatically.

```bash
python 1_extract_dictionary_text.py dictionary_20260105.pdf --start-page 16
```

**Output**: `outputs/extractions/extraction_TIMESTAMP.txt`

### 2. Parse (Make it structured)

Feed the raw text to the model to identify Headwords, Definitions, and Examples.

```bash
python 2_parse_dictionary_entries.py
```

**Output**: `outputs/parsed/entries_TIMESTAMP.json`

### 3. Clean (Scrub the artifacts)

Standardize placeholders and fix the "OCR jitter."

```bash
python 3_clean_placeholders.py outputs/parsed/entries_TIMESTAMP.json

# Optional: Specialized dash cleaning
python 3.1_replace_dash_space_dash.py outputs/cleaned/progress_TIMESTAMP.json
```

**Output**: `outputs/cleaned/progress_TIMESTAMP_original_cleaned.json`

### 4. Correct (Fix typos)

Corrects OCR slips while maintaining the integrity of the text.

```bash
python 4_correct_ocr_typos.py outputs/cleaned/progress_TIMESTAMP.json
```

**Output**: `outputs/corrected/corrections_TIMESTAMP.json`

### 5. Extract (Build the dataset)

Generate the final gold standard: parallel sentences for training.

```bash
python 5_extract_parallel_sentences.py outputs/cleaned/progress_TIMESTAMP.json --count 100
```

**Output**: `outputs/parallel_sentences_100.jsonl`

**The Result:**

```json
{"ambonese": "Beta pigi ka pasar", "indonesian": "Saya pergi ke pasar"}
```

## 📂 Project Structure

```
project-manise/
├── 1_extract_dictionary_text.py      # The Extractor
├── 2_parse_dictionary_entries.py     # The Parser
├── 3_clean_placeholders.py           # The Cleaner
├── 4_correct_ocr_typos.py            # The Corrector
├── 5_extract_parallel_sentences.py   # The Miner
├── convert_json_csv.py               # Converter (JSON <-> CSV)
├── utils/                            # Utilities
│   ├── change_tracker.py
│   ├── logger.py
│   └── validators.py
└── outputs/                          # Generated Data
    └── parallel_sentences_100.jsonl
```

## 📊 Data Formats

### The Nested JSON (Rich Dictionary Data):

```json
{
  "headword": "pigi",
  "meanings": [
    {
      "definition": "pergi",
      "ambonese_example": "Beta pigi ka pasar",
      "indonesian_translation": "Saya pergi ke pasar"
    }
  ]
}
```

### The Flat JSONL (Ready for Training):

```json
{"ambonese": "Beta pigi ka pasar", "indonesian": "Saya pergi ke pasar"}
```

## 📝 Notes

- **Resume Capability**: Scripts track progress and can resume where they left off if interrupted.
- **Logs**: Detailed logs are saved to `outputs/logs/`.
- **Cost Warning**: This pipeline utilizes API calls. Monitor usage when processing large dictionaries.

Built for the documentation and preservation of Ambonese Malay.
