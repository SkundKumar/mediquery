import os
import json
import pandas as pd
from lxml import etree
from tqdm import tqdm

def parse_medquad_xml(file_path):
    data_points = []
    try:
        tree = etree.parse(file_path)
        root = tree.getroot()
        for qa in root.xpath('.//QAPair'):
            question = qa.findtext('Question')
            answer = qa.findtext('Answer')
            if question and answer:
                combined_text = f"Question: {question.strip()}\nAnswer: {answer.strip()}"
                data_points.append({"text": combined_text, "metadata": {"source": "MedQuAD", "file": os.path.basename(file_path)}})
    except Exception as e:
        print(f"Error parsing XML {file_path}: {e}")
    return data_points

def parse_medquad_csv(file_path):
    data_points = []
    try:
        # MedQuAD CSV usually has 'Question' and 'Answer' columns
        df = pd.read_csv(file_path)
        # Normalize column names to handle case sensitivity
        df.columns = [c.strip().capitalize() for c in df.columns]
        
        if 'Question' in df.columns and 'Answer' in df.columns:
            for _, row in df.iterrows():
                question = str(row['Question'])
                answer = str(row['Answer'])
                combined_text = f"Question: {question.strip()}\nAnswer: {answer.strip()}"
                data_points.append({"text": combined_text, "metadata": {"source": "MedQuAD", "file": os.path.basename(file_path)}})
        else:
            print(f"Columns not found in {file_path}. Found: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error parsing CSV {file_path}: {e}")
    return data_points

def main():
    raw_dir = 'data/raw'
    processed_dir = 'data/processed'
    output_file = os.path.join(processed_dir, 'med_data_cleaned.jsonl')
    all_data = []
    
    # Now looking for BOTH .xml and .csv
    files = [f for f in os.listdir(raw_dir) if f.endswith(('.xml', '.csv'))]
    print(f"Found {len(files)} total files. Starting pre-processing...")

    for filename in tqdm(files):
        file_path = os.path.join(raw_dir, filename)
        if filename.endswith('.xml'):
            all_data.extend(parse_medquad_xml(file_path))
        else:
            all_data.extend(parse_medquad_csv(file_path))

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_data:
            f.write(json.dumps(entry) + '\n')

    print(f"Successfully processed {len(all_data)} medical chunks.")
    print(f"Cleaned data saved to: {output_file}")

if __name__ == "__main__":
    main()