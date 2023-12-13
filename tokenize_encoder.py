import os
import re
from transformers import AutoTokenizer


class ClinicalTextProcessor:
    def __init__(self, study_id_list, folder_paths):
        self.study_id_list = study_id_list
        self.folder_paths = folder_paths
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    def process_files(self):
        for folder_path in self.folder_paths:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.txt'):
                        file_id = file.split('.')[0][1:]
                        if file_id in self.study_id_list:
                            self.process_file(os.path.join(root, file))

    def process_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as text_file:
            text_content = text_file.read()

        findings_text, impression_text = self.extract_sections(text_content)

        findings_inputs = self.tokenize_text(findings_text)
        impression_inputs = self.tokenize_text(impression_text)

        print(f"Processed {file_path}")  #

        """need to +a code..."""

    def extract_sections(self, text):
        findings_match = re.search(r"FINDINGS:(.*?)IMPRESSION:", text, re.S)
        impression_match = re.search(r"IMPRESSION:(.*)", text, re.S)

        findings_text = findings_match.group(1).strip() if findings_match else ""
        impression_text = impression_match.group(1).strip() if impression_match else ""
        return findings_text, impression_text

    def tokenize_text(self, text):
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)


# class example
study_id_list = ['your', 'list', 'of', 'ids']  # study list
folder_paths = [f'C:\\your_path\\p{num}' for num in range(10, 20)]  # your path
processor = ClinicalTextProcessor(study_id_list, folder_paths)
processor.process_files()
