---
annotations_creators:
- human-annotated
language:
- vie
license: mit
multilinguality: monolingual
source_datasets:
- GreenNode/GreenNode-Table-Markdown-Retrieval-VN
task_categories:
- text-retrieval
task_ids:
- document-retrieval
dataset_info:
- config_name: corpus
  features:
  - name: id
    dtype: string
  - name: title
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: test
    num_bytes: 70641120
    num_examples: 44678
  download_size: 32478180
  dataset_size: 70641120
- config_name: default
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: float64
  splits:
  - name: train
    num_bytes: 12879540
    num_examples: 143106
  - name: test
    num_bytes: 3221190
    num_examples: 35791
- config_name: qrels
  features:
  - name: query-id
    dtype: string
  - name: corpus-id
    dtype: string
  - name: score
    dtype: int64
  splits:
  - name: test
    num_bytes: 3221190
    num_examples: 35791
  download_size: 1667183
  dataset_size: 3221190
- config_name: queries
  features:
  - name: id
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: test
    num_bytes: 5768313
    num_examples: 35791
  download_size: 2417145
  dataset_size: 5768313
configs:
- config_name: corpus
  data_files:
  - split: test
    path: corpus/test-*
- config_name: default
  data_files:
  - split: test
    path: qrels/test.jsonl
  - split: train
    path: qrels/train.jsonl
- config_name: qrels
  data_files:
  - split: test
    path: qrels/test-*
- config_name: queries
  data_files:
  - split: test
    path: queries/test-*
tags:
- mteb
- text
---
<!-- adapted from https://github.com/huggingface/huggingface_hub/blob/v0.30.2/src/huggingface_hub/templates/datasetcard_template.md -->

<div align="center" style="padding: 40px 20px; background-color: white; border-radius: 12px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05); max-width: 600px; margin: 0 auto;">
  <h1 style="font-size: 3.5rem; color: #1a1a1a; margin: 0 0 20px 0; letter-spacing: 2px; font-weight: 700;">GreenNodeTableMarkdownRetrieval</h1>
  <div style="font-size: 1.5rem; color: #4a4a4a; margin-bottom: 5px; font-weight: 300;">An <a href="https://github.com/embeddings-benchmark/mteb" style="color: #2c5282; font-weight: 600; text-decoration: none;" onmouseover="this.style.textDecoration='underline'" onmouseout="this.style.textDecoration='none'">MTEB</a> dataset</div>
  <div style="font-size: 0.9rem; color: #2c5282; margin-top: 10px;">Massive Text Embedding Benchmark</div>
</div>

GreenNodeTable documents

|               |                                             |
|---------------|---------------------------------------------|
| Task category | t2t                              |
| Domains       | Financial, Encyclopaedic, Non-fiction                               |
| Reference     | https://huggingface.co/GreenNode |

Source datasets:
- [GreenNode/GreenNode-Table-Markdown-Retrieval-VN](https://huggingface.co/datasets/GreenNode/GreenNode-Table-Markdown-Retrieval-VN)


## How to evaluate on this task

You can evaluate an embedding model on this dataset using the following code:

```python
import mteb

task = mteb.get_task("GreenNodeTableMarkdownRetrieval")
evaluator = mteb.MTEB([task])

model = mteb.get_model(YOUR_MODEL)
evaluator.run(model)
```

<!-- Datasets want link to arxiv in readme to autolink dataset with paper -->
To learn more about how to run models on `mteb` task check out the [GitHub repository](https://github.com/embeddings-benchmark/mteb).

## Citation

If you use this dataset, please cite the dataset as well as [mteb](https://github.com/embeddings-benchmark/mteb), as this dataset likely includes additional processing as a part of the [MMTEB Contribution](https://github.com/embeddings-benchmark/mteb/tree/main/docs/mmteb).

```bibtex

@inproceedings{10.1007/978-981-95-1746-6_17,
  abstract = {Information retrieval often comes in plain text, lacking semi-structured text such as HTML and markdown, retrieving data that contains rich format such as table became non-trivial. In this paper, we tackle this challenge by introducing a new dataset, GreenNode Table Retrieval VN (GN-TRVN), which is collected from a massive corpus, a wide range of topics, and a longer context compared to ViQuAD2.0. To evaluate the effectiveness of our proposed dataset, we introduce two versions, M3-GN-VN and M3-GN-VN-Mixed, by fine-tuning the M3-Embedding model on this dataset. Experimental results show that our models consistently outperform the baselines, including the base model, across most evaluation criteria on various datasets such as VieQuADRetrieval, ZacLegalTextRetrieval, and GN-TRVN. In general, we release a more comprehensive dataset and two model versions that improve response performance for Vietnamese Markdown Table Retrieval.},
  address = {Singapore},
  author = {Pham, Bao Loc
and Hoang, Quoc Viet
and Luu, Quy Tung
and Vo, Trong Thu},
  booktitle = {Proceedings of the Fifth International Conference on Intelligent Systems and Networks},
  isbn = {978-981-95-1746-6},
  pages = {153--163},
  publisher = {Springer Nature Singapore},
  title = {GN-TRVN: A Benchmark for Vietnamese Table Markdown Retrieval Task},
  year = {2026},
}


@article{enevoldsen2025mmtebmassivemultilingualtext,
  title={MMTEB: Massive Multilingual Text Embedding Benchmark},
  author={Kenneth Enevoldsen and Isaac Chung and Imene Kerboua and Márton Kardos and Ashwin Mathur and David Stap and Jay Gala and Wissam Siblini and Dominik Krzemiński and Genta Indra Winata and Saba Sturua and Saiteja Utpala and Mathieu Ciancone and Marion Schaeffer and Gabriel Sequeira and Diganta Misra and Shreeya Dhakal and Jonathan Rystrøm and Roman Solomatin and Ömer Çağatan and Akash Kundu and Martin Bernstorff and Shitao Xiao and Akshita Sukhlecha and Bhavish Pahwa and Rafał Poświata and Kranthi Kiran GV and Shawon Ashraf and Daniel Auras and Björn Plüster and Jan Philipp Harries and Loïc Magne and Isabelle Mohr and Mariya Hendriksen and Dawei Zhu and Hippolyte Gisserot-Boukhlef and Tom Aarsen and Jan Kostkan and Konrad Wojtasik and Taemin Lee and Marek Šuppa and Crystina Zhang and Roberta Rocca and Mohammed Hamdy and Andrianos Michail and John Yang and Manuel Faysse and Aleksei Vatolin and Nandan Thakur and Manan Dey and Dipam Vasani and Pranjal Chitale and Simone Tedeschi and Nguyen Tai and Artem Snegirev and Michael Günther and Mengzhou Xia and Weijia Shi and Xing Han Lù and Jordan Clive and Gayatri Krishnakumar and Anna Maksimova and Silvan Wehrli and Maria Tikhonova and Henil Panchal and Aleksandr Abramov and Malte Ostendorff and Zheng Liu and Simon Clematide and Lester James Miranda and Alena Fenogenova and Guangyu Song and Ruqiya Bin Safi and Wen-Ding Li and Alessia Borghini and Federico Cassano and Hongjin Su and Jimmy Lin and Howard Yen and Lasse Hansen and Sara Hooker and Chenghao Xiao and Vaibhav Adlakha and Orion Weller and Siva Reddy and Niklas Muennighoff},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2502.13595},
  year={2025},
  url={https://arxiv.org/abs/2502.13595},
  doi = {10.48550/arXiv.2502.13595},
}

@article{muennighoff2022mteb,
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Loïc and Reimers, Nils},
  title = {MTEB: Massive Text Embedding Benchmark},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2210.07316},
  year = {2022}
  url = {https://arxiv.org/abs/2210.07316},
  doi = {10.48550/ARXIV.2210.07316},
}
```

# Dataset Statistics
<details>
  <summary> Dataset Statistics</summary>

The following code contains the descriptive statistics from the task. These can also be obtained using:

```python
import mteb

task = mteb.get_task("GreenNodeTableMarkdownRetrieval")

desc_stats = task.metadata.descriptive_stats
```

```json
{
    "test": {
        "num_samples": 80469,
        "number_of_characters": 59810147,
        "documents_text_statistics": {
            "total_text_length": 56678343,
            "min_text_length": 74,
            "average_text_length": 1268.596244236537,
            "max_text_length": 4074,
            "unique_texts": 44678
        },
        "documents_image_statistics": null,
        "queries_text_statistics": {
            "total_text_length": 3131804,
            "min_text_length": 3,
            "average_text_length": 87.50255650861949,
            "max_text_length": 337,
            "unique_texts": 35554
        },
        "queries_image_statistics": null,
        "relevant_docs_statistics": {
            "num_relevant_docs": 35791,
            "min_relevant_docs_per_query": 1,
            "average_relevant_docs_per_query": 1.0,
            "max_relevant_docs_per_query": 1,
            "unique_relevant_docs": 8936
        },
        "top_ranked_statistics": null
    }
}
```

</details>

---
*This dataset card was automatically generated using [MTEB](https://github.com/embeddings-benchmark/mteb)*