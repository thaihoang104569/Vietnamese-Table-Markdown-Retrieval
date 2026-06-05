import re
from underthesea import word_tokenize

def tokenize_vietnamese(text: str) -> str:
    """
    Tokenize Vietnamese text, replacing spaces in compound words with underscores.
    This function is table-aware: it tokenizes cell-by-cell inside markdown tables
    to preserve the table structure and pipeline alignment.
    """
    if not text:
        return ""
        
    lines = text.split('\n')
    tokenized_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            tokenized_lines.append(line)
            continue
            
        # Check if the line is part of a markdown table (starts/ends with | or contains multiple |)
        if stripped.startswith('|') or (len(re.findall(r'\|', line)) >= 2):
            cells = line.split('|')
            tokenized_cells = []
            for cell in cells:
                # Preserve empty or padding cells
                if not cell.strip():
                    tokenized_cells.append(cell)
                else:
                    # Tokenize content inside the cell and join compound words with underscores
                    cell_content = cell.strip()
                    tokenized_content = word_tokenize(cell_content, format="text")
                    # Match the spacing style of the original cell
                    tokenized_cells.append(f" {tokenized_content} ")
            tokenized_lines.append('|'.join(tokenized_cells))
        else:
            # Tokenize regular text line
            tokenized_lines.append(word_tokenize(line, format="text"))
            
    return '\n'.join(tokenized_lines)

def tokenize_query(query: str) -> str:
    """
    Tokenize a search query using Vietnamese word segmentation.
    """
    if not query:
        return ""
    return word_tokenize(query, format="text")
