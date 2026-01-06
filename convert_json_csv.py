#!/usr/bin/env python3
"""
Convert JSON dictionary data to CSV and back.

This script handles the conversion between JSON format (with nested meanings arrays)
and CSV format (with flattened rows, one per meaning).

Usage:
    # Convert JSON to CSV
    python convert_json_csv.py json_to_csv input.json output.csv

    # Convert CSV to JSON
    python convert_json_csv.py csv_to_json input.csv output.json

    # Convert CSV to JSONL (one entry per line)
    python convert_json_csv.py csv_to_jsonl input.csv output.jsonl
"""

import json
import csv
import sys
import argparse
from pathlib import Path
from collections import defaultdict


def json_to_csv(json_file, csv_file):
    """Convert JSON dictionary data to CSV format."""
    print(f"Reading JSON from: {json_file}")
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract metadata and entries
    metadata = {k: v for k, v in data.items() if k != 'entries'}
    entries = data.get('entries', [])
    
    print(f"Found {len(entries)} entries")
    
    # Prepare CSV rows - flatten meanings
    rows = []
    for entry in entries:
        headword = entry.get('headword', '')
        page_number = entry.get('page_number', '')
        
        meanings = entry.get('meanings', [])
        if not meanings:
            # If no meanings, create one row with empty meaning fields
            rows.append({
                'headword': headword,
                'page_number': page_number,
                'sense_number': '',
                'definition': '',
                'ambonese_example': '',
                'indonesian_translation': ''
            })
        else:
            for meaning in meanings:
                rows.append({
                    'headword': headword,
                    'page_number': page_number,
                    'sense_number': meaning.get('sense_number', ''),
                    'definition': meaning.get('definition', ''),
                    'ambonese_example': meaning.get('ambonese_example', ''),
                    'indonesian_translation': meaning.get('indonesian_translation', '')
                })
    
    # Write CSV
    fieldnames = ['headword', 'page_number', 'sense_number', 'definition', 'ambonese_example', 'indonesian_translation']
    
    print(f"Writing {len(rows)} rows to CSV: {csv_file}")
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    
    # Save metadata to a separate file for reference
    metadata_file = csv_file.replace('.csv', '_metadata.json')
    print(f"Saving metadata to: {metadata_file}")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Conversion complete! {len(rows)} rows written to CSV")


def csv_to_json(csv_file, json_file, metadata_file=None):
    """Convert CSV back to JSON format, reconstructing nested meanings."""
    print(f"Reading CSV from: {csv_file}")
    
    rows = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    print(f"Found {len(rows)} rows")
    
    # Group rows by headword and page_number to reconstruct entries
    entries_dict = defaultdict(lambda: {
        'meanings': []
    })
    
    for row in rows:
        headword = row.get('headword', '').strip()
        page_number = row.get('page_number', '').strip()
        
        # Use headword + page_number as key
        key = (headword, page_number)
        
        if headword:
            entries_dict[key]['headword'] = headword
            entries_dict[key]['page_number'] = int(page_number) if page_number.isdigit() else page_number
            
            # Add meaning if definition or other fields are present
            sense_num = row.get('sense_number', '').strip()
            definition = row.get('definition', '').strip()
            ambonese_example = row.get('ambonese_example', '').strip()
            indonesian_translation = row.get('indonesian_translation', '').strip()
            
            if sense_num or definition or ambonese_example or indonesian_translation:
                meaning = {
                    'sense_number': int(sense_num) if sense_num.isdigit() else sense_num,
                    'definition': definition,
                    'ambonese_example': ambonese_example,
                    'indonesian_translation': indonesian_translation
                }
                entries_dict[key]['meanings'].append(meaning)
    
    # Convert to list of entries
    entries = list(entries_dict.values())
    
    # Load metadata if provided
    metadata = {}
    if metadata_file and Path(metadata_file).exists():
        print(f"Loading metadata from: {metadata_file}")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        # Try to find metadata file automatically
        auto_metadata_file = csv_file.replace('.csv', '_metadata.json')
        if Path(auto_metadata_file).exists():
            print(f"Loading metadata from: {auto_metadata_file}")
            with open(auto_metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
    
    # Combine metadata and entries
    output_data = {**metadata, 'entries': entries}
    
    print(f"Writing {len(entries)} entries to JSON: {json_file}")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Conversion complete! {len(entries)} entries written to JSON")


def csv_to_jsonl(csv_file, jsonl_file, metadata_file=None):
    """Convert CSV to JSONL format (one entry per line)."""
    print(f"Reading CSV from: {csv_file}")
    
    rows = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    print(f"Found {len(rows)} rows")
    
    # Group rows by headword and page_number to reconstruct entries
    entries_dict = defaultdict(lambda: {
        'meanings': []
    })
    
    for row in rows:
        headword = row.get('headword', '').strip()
        page_number = row.get('page_number', '').strip()
        
        key = (headword, page_number)
        
        if headword:
            entries_dict[key]['headword'] = headword
            entries_dict[key]['page_number'] = int(page_number) if page_number.isdigit() else page_number
            
            sense_num = row.get('sense_number', '').strip()
            definition = row.get('definition', '').strip()
            ambonese_example = row.get('ambonese_example', '').strip()
            indonesian_translation = row.get('indonesian_translation', '').strip()
            
            if sense_num or definition or ambonese_example or indonesian_translation:
                meaning = {
                    'sense_number': int(sense_num) if sense_num.isdigit() else sense_num,
                    'definition': definition,
                    'ambonese_example': ambonese_example,
                    'indonesian_translation': indonesian_translation
                }
                entries_dict[key]['meanings'].append(meaning)
    
    entries = list(entries_dict.values())
    
    print(f"Writing {len(entries)} entries to JSONL: {jsonl_file}")
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✓ Conversion complete! {len(entries)} entries written to JSONL")


def main():
    parser = argparse.ArgumentParser(
        description='Convert between JSON dictionary format and CSV format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert JSON to CSV
  python convert_json_csv.py json_to_csv input.json output.csv

  # Convert CSV back to JSON
  python convert_json_csv.py csv_to_json input.csv output.json

  # Convert CSV to JSONL
  python convert_json_csv.py csv_to_jsonl input.csv output.jsonl
        """
    )
    
    parser.add_argument('command', choices=['json_to_csv', 'csv_to_json', 'csv_to_jsonl'],
                       help='Conversion direction')
    parser.add_argument('input_file', help='Input file path')
    parser.add_argument('output_file', help='Output file path')
    parser.add_argument('--metadata', help='Metadata JSON file (for csv_to_json)')
    
    args = parser.parse_args()
    
    if args.command == 'json_to_csv':
        json_to_csv(args.input_file, args.output_file)
    elif args.command == 'csv_to_json':
        csv_to_json(args.input_file, args.output_file, args.metadata)
    elif args.command == 'csv_to_jsonl':
        csv_to_jsonl(args.input_file, args.output_file, args.metadata)


if __name__ == '__main__':
    main()

