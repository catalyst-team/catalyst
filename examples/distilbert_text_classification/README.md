## Transformers for sequence classification

### Prerequisites
You need to install developer version of catalyst (`pip install -e .` in [catalyst](https://github.com/catalyst-team/catalyst) main folder).

### Running text classification experiment with DistilBert
1. Specify path to data and train/validation file names in `config.yaml`. For instance, you can [download](https://www.kaggle.com/c/amazon-pet-product-reviews-classification/data) Amazon pet product reviews and put training and validation files into the `input` folder
2. Specify text and label columns to read. By default, these are `text` for text field and `label` for targets
3. From the `examples` folder, run `catalyst-dl run -C distilbert_text_classification/config.yml`
4. When training is done, you'll find logs, configs and model checkpoints in the `logs` folder (this path is also configurable)  

Same with Jupyter API - [tutorial](https://www.kaggle.com/kashnitsky/distillbert-catalyst-amazon-product-reviews)