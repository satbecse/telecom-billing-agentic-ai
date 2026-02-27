import os
import glob
from fpdf import FPDF

def convert_txt_to_pdf():
    txt_files = glob.glob('data/docs/*.txt')
    os.makedirs('data/docs/customer_pdfs', exist_ok=True)
    
    for txt_file in txt_files:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Courier", size=10)
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Basic encoding handling for FPDF
                clean_line = line.encode('latin-1', 'replace').decode('latin-1')
                pdf.cell(0, 5, txt=clean_line, new_x="LMARGIN", new_y="NEXT")
                
        base_name = os.path.basename(txt_file).replace('.txt', '.pdf')
        out_path = os.path.join('data/docs/customer_pdfs', base_name)
        pdf.output(out_path)
        print(f"Converted {base_name}")

if __name__ == "__main__":
    convert_txt_to_pdf()
