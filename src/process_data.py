import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

def main():
    raw_dir = "src/Data/raw"
    processed_dir = "src/Data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(os.path.join(processed_dir, "qrels"), exist_ok=True)

    corpus_raw_path = os.path.join(raw_dir, "corpus.jsonl")
    queries_raw_path = os.path.join(raw_dir, "queries.jsonl")
    qrels_train_raw_path = os.path.join(raw_dir, "qrels", "train.jsonl")
    qrels_test_raw_path = os.path.join(raw_dir, "qrels", "test.jsonl")

    corpus_proc_path = os.path.join(processed_dir, "corpus.jsonl")
    queries_proc_path = os.path.join(processed_dir, "queries.jsonl")
    qrels_train_proc_path = os.path.join(processed_dir, "qrels", "train.jsonl")
    qrels_test_proc_path = os.path.join(processed_dir, "qrels", "test.jsonl")

    # 1. Process Queries: filter empty/whitespace queries
    print("Processing queries...")
    valid_query_ids = set()
    query_count = 0
    filtered_query_count = 0
    with open(queries_raw_path, "r", encoding="utf-8") as fin, \
         open(queries_proc_path, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            q_id = item.get("_id")
            text = item.get("text", "")
            
            # Check if query is empty or only whitespace
            if not text.strip():
                filtered_query_count += 1
                continue
                
            # Keep only relevant fields
            cleaned_item = {
                "_id": q_id,
                "text": text.strip()
            }
            fout.write(json.dumps(cleaned_item, ensure_ascii=False) + "\n")
            valid_query_ids.add(q_id)
            query_count += 1
            
    print(f"Queries: saved {query_count} valid queries, filtered out {filtered_query_count} empty queries.")

    # 2. Process Corpus: remove 'title' field since all are "None"
    print("Processing corpus...")
    valid_doc_ids = set()
    doc_count = 0
    with open(corpus_raw_path, "r", encoding="utf-8") as fin, \
         open(corpus_proc_path, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            d_id = item.get("_id")
            text = item.get("text", "")
            
            # Keep only doc_id and text, ignoring title
            cleaned_item = {
                "_id": d_id,
                "text": text.strip()
            }
            fout.write(json.dumps(cleaned_item, ensure_ascii=False) + "\n")
            valid_doc_ids.add(d_id)
            doc_count += 1
            
    print(f"Corpus: saved {doc_count} documents (title field discarded).")

    # 3. Process Train Qrels: filter mapping for removed queries
    print("Processing train qrels...")
    train_qrels_count = 0
    train_filtered_count = 0
    with open(qrels_train_raw_path, "r", encoding="utf-8") as fin, \
         open(qrels_train_proc_path, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            q_id = item.get("query-id")
            d_id = item.get("corpus-id")
            score = item.get("score", 1)
            
            if q_id not in valid_query_ids or d_id not in valid_doc_ids:
                train_filtered_count += 1
                continue
                
            cleaned_item = {
                "query-id": q_id,
                "corpus-id": d_id,
                "score": score
            }
            fout.write(json.dumps(cleaned_item, ensure_ascii=False) + "\n")
            train_qrels_count += 1
            
    print(f"Train Qrels: saved {train_qrels_count} relations, filtered out {train_filtered_count} invalid relations.")

    # 4. Process Test Qrels: filter mapping for removed queries
    print("Processing test qrels...")
    test_qrels_count = 0
    test_filtered_count = 0
    with open(qrels_test_raw_path, "r", encoding="utf-8") as fin, \
         open(qrels_test_proc_path, "w", encoding="utf-8") as fout:
        for line in fin:
            item = json.loads(line)
            q_id = item.get("query-id")
            d_id = item.get("corpus-id")
            score = item.get("score", 1)
            
            if q_id not in valid_query_ids or d_id not in valid_doc_ids:
                test_filtered_count += 1
                continue
                
            cleaned_item = {
                "query-id": q_id,
                "corpus-id": d_id,
                "score": score
            }
            fout.write(json.dumps(cleaned_item, ensure_ascii=False) + "\n")
            test_qrels_count += 1
            
    print(f"Test Qrels: saved {test_qrels_count} relations, filtered out {test_filtered_count} invalid relations.")
    print("Data processing finished successfully.")

if __name__ == "__main__":
    main()
