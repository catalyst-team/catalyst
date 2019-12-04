## Transformers for sequence classification

### Prerequisites
`pip install catalyst-nlp`

### Running text classification experiment with DistilBert
1. Specify paths to train and validation csv files in `config.yaml`
2. Specify text and label columns to read. By default, these are `text` for text field and `label` for targets
3. From the `examples` folder, run `catalyst-dl run -C nlp_bert_classification/config.yml`
4. When training is done, you'll find logs, configs and model checkpoints in the `logs` folder (this path is also configurable)  