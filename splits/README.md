# Data splits and checksums

Row indices of the Megadiff train / validation / test splits used in this study (RESEARCH_LOG #39, #49). A row index is the position of a pair in the Hugging Face dataset `ASSERT-KTH/megadiff-single-function` (split `train`). The code itself isn't redistributed; `src/data_loader.py` rebuilds the files deterministically (seed 42, max_length 768, split by file path).

| File | Contents |
|---|---|
| `megadiff_train_rows.txt` | 10,000 row indices |
| `megadiff_validation_rows.txt` | 1,000 row indices |
| `megadiff_test_rows.txt` | 1,000 row indices |

SHA-256 of the generated data files, to check that a rebuild is byte-identical:

| Generated file | Records | SHA-256 |
|---|---|---|
| `data/megadiff_train.jsonl` | 10,000 | `5f7ff458c92a25d6eb83ad8abbb39683a06a22396b5cb096f9b7ebc2ce5ad60a` |
| `data/megadiff_validation.jsonl` | 1,000 | `1ab24133bbd6426bf327689af439e1f57ce0ad7dce0d2853e2be71a3f131e3cd` |
| `data/megadiff_test.jsonl` | 1,000 | `947ab2f4167fb4efd8b2086a8e49b9a7370dd674f63f61411ae7cfe84b7f47cc` |
| `data/humanevalfix_java.jsonl` | 164 | `b00d28ffb97863c552c2f021e9ab8eb36f0655c15737f2797972f2f6b7e4c9a0` |
| `data/humaneval_java.jsonl` | 163 | `4b0e801caf7992ec79e32ab845e2aa85590557271a99b522ef6b27211df4e2c6` |
