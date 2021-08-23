# Improving Czech Sentiment with Transformers
Repository for the paper:
##Are the Multilingual Models Better? Improving Czech Sentiment with Transformers

If you use this software for academic research, [please cite the paper](#publication)


Usage:
--------
Logistic Regression Baseline
```
python3 baseline.py --dataset_name fb --use_train_test --word_ngram_vectorizer cv --char_ngram_vectorizer cv
python3 baseline.py --dataset_name fb --use_train_test --word_ngram_vectorizer cv --char_ngram_vectorizer cv --binary

python3 baseline.py --dataset_name csfd --use_train_test --word_ngram_vectorizer cv --char_ngram_vectorizer none
python3 baseline.py --dataset_name csfd --use_train_test --word_ngram_vectorizer cv --char_ngram_vectorizer none --binary

python3 baseline.py --dataset_name mallcz --use_train_test --word_ngram_vectorizer cv --char_ngram_vectorizer none
python3 baseline.py --dataset_name mallcz --use_train_test --word_ngram_vectorizer cv --char_ngram_vectorizer none --binary

```


LSTM Baseline
you have to delete the temporary directory `/data/lstm_baseline` before each run 

```
python3 run_lstm_baseline.py --dataset_name fb --model_name lstm-base --embeddings_file cc.cs.300.vec --embeddings_size 300 --max_words 300000 --max_seq_len 64 --trainable_word_embeddings --tokenizer toktok --use_stemmer --use_data_cleaner --max_seq_len 512 --num_repeat 1 --use_attention --attention_type multiplicative --rnn_cells 256 --epoch_count 5 --optimizer AdamW --batch_size 256 --lr 0.0005 --lr_scheduler_name cosine --warm_up_steps 0.1 --use_only_train_data --weight_decay 0.0001
python3 run_lstm_baseline.py --dataset_name fb --model_name lstm-base --embeddings_file cc.cs.300.vec --embeddings_size 300 --max_words 300000 --max_seq_len 64 --trainable_word_embeddings --tokenizer toktok --use_stemmer --use_data_cleaner --max_seq_len 512 --num_repeat 1 --use_attention --attention_type multiplicative --rnn_cells 256 --epoch_count 5 --optimizer AdamW --batch_size 256 --lr 0.0005 --lr_scheduler_name cosine --warm_up_steps 0.1 --use_only_train_data --weight_decay 0.0001 --binary


python3 run_lstm_baseline.py --dataset_name csfd --model_name lstm-base --embeddings_file cc.cs.300.vec --embeddings_size 300 --max_words 300000 --trainable_word_embeddings --tokenizer toktok --use_stemmer --use_data_cleaner --max_seq_len 512 --num_repeat 1 --use_attention --attention_type multiplicative --rnn_cells 128 --epoch_count 2 --optimizer AdamW --batch_size 128 --lr 0.0005 --lr_scheduler_name cosine --warm_up_steps 0.1 --use_only_train_data
python3 run_lstm_baseline.py --dataset_name csfd --model_name lstm-base --embeddings_file cc.cs.300.vec --embeddings_size 300 --max_words 300000 --trainable_word_embeddings --tokenizer toktok --use_stemmer --use_data_cleaner --max_seq_len 512 --num_repeat 1 --use_attention --attention_type multiplicative --rnn_cells 128 --epoch_count 2 --optimizer AdamW --batch_size 128 --lr 0.0005 --lr_scheduler_name cosine --warm_up_steps 0.1 --use_only_train_data --binary

python3 run_lstm_baseline.py --dataset_name mallcz --model_name lstm-base --embeddings_file cc.cs.300.vec --embeddings_size 300 --max_words 300000 --trainable_word_embeddings --tokenizer toktok --use_stemmer --use_data_cleaner --max_seq_len 512 --num_repeat 1 --use_attention --attention_type multiplicative --rnn_cells 128 --epoch_count 10 --optimizer AdamW --batch_size 128 --lr 0.0005 --lr_scheduler_name cosine --warm_up_steps 0.1 --use_only_train_data
python3 run_lstm_baseline.py --dataset_name mallcz --model_name lstm-base --embeddings_file cc.cs.300.vec --embeddings_size 300 --max_words 300000 --trainable_word_embeddings --tokenizer toktok --use_stemmer --use_data_cleaner --max_seq_len 512 --num_repeat 1 --use_attention --attention_type multiplicative --rnn_cells 128 --epoch_count 2 --optimizer AdamW --batch_size 128 --lr 0.0005 --lr_scheduler_name cosine --warm_up_steps 0.1 --use_only_train_data --binary

```

Transformers Fine-Tuning

for example CSFD 
```
# CzechALBERT
python3 run_polarity_fine_tuning.py --model_name ./data/pre-trained/czert-base-uncased --dataset_name csfd --batch_size 14 --from_tf --tokenizer_type berttokenizerfast --lr 2e-6 --epoch_num 8 --use_only_train_data --model_type albert --use_random_seed --max_seq_len 512

# CzechBERT
python3 run_polarity_fine_tuning.py --model_name ./data/pre-trained/czert-bert-base-cased --dataset_name csfd --batch_size 14 --from_tf --tokenizer_type berttokenizerfast --lr 2e-5 --epoch_num 12 --use_only_train_data --use_random_seed --max_seq_len 512

# mBERT
python3 run_polarity_fine_tuning.py --model_name bert-base-multilingual-cased --dataset_name csfd --batch_size 14 --lr 2e-6 --epoch_num 13 --use_only_train_data --use_random_seed --max_seq_len 512

# SlavicBERT
python3 run_polarity_fine_tuning.py --model_name DeepPavlov/bert-base-bg-cs-pl-ru-cased --dataset_name csfd --batch_size 14 --lr 2e-6 --epoch_num 12 --use_only_train_data --use_random_seed --max_seq_len 512

# RandomALBERT
python3 run_polarity_fine_tuning.py --model_name ./data/pre-trained/base_random --dataset_name csfd --batch_size 14 --from_tf --tokenizer_type berttokenizerfast --lr 2e-6 --epoch_num 14 --model_type albert --use_random_seed --max_seq_len 512 --use_only_train_data

# XLM-R-Base
python3 run_polarity_fine_tuning.py --model_type xlm-r --model_name xlm-roberta-base --dataset_name csfd --batch_size 32 --tokenizer_type xlm-r-tokenizer --scheduler constant --lr 2e-6 --epoch_num 15 --use_random_seed --max_seq_len 512 --use_only_train_data --enable_wandb --num_repeat 1

# XLM-R-Large
python3 run_polarity_fine_tuning.py --model_type xlm-r --model_name xlm-roberta-large --dataset_name csfd --batch_size 28 --tokenizer_type xlm-r-tokenizer --lr 2e-6 --epoch_num 11 --use_random_seed --max_seq_len 512 --use_only_train_data --data_parallel --enable_wandb --num_repeat 1

# XLM
python3 run_polarity_fine_tuning.py --model_type xlm --model_name xlm-mlm-100-1280 --dataset_name csfd --batch_size 34 --tokenizer_type xlmtokenizer --lr 2e-5 --epoch_num 11 --use_random_seed --max_seq_len 512 --use_only_train_data --enable_wandb --num_repeat 1 --data_parallel

```

Setup:
--------

Create conda enviroment

1) #### Clone github repository 
   ```
   git clone ...
   ```
2) #### Setup conda
    Check version
    ```
    # print version
    conda -V
   
    # print available enviroments
    conda info --envs
    ```
    Create conda enviroment
   
    ```
    # create enviroment 
    conda create --name improving-czech-sentiment-transformers python=3.7 -y
    
    # check that it was created
    conda info --envs
   
    # activate enviroment
    conda activate improving-czech-sentiment-transformers
   
    # see already installed packages
    pip freeze  
    ```
   
   Install requirements
   ```
   pip install -r requirements.txt
   ```
   
   PyTorch with cuda must be installed as well
   ```
   pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
   pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 torchtext==0.6.0 -f https://download.pytorch.org/whl/torch_stable.html
   ```
   or
   ```
   conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
   ```
3) #### Setup Data
   Run the init script `init_folders.py` 
    ```
    python3 init_folders.py
    ```

   Download the [Czech datasets](http://nlp.kiv.zcu.cz/research/sentiment) ([CSFD](http://nlp.kiv.zcu.cz/data/research/sentiment/csfd.zip), [FB](http://nlp.kiv.zcu.cz/data/research/sentiment/facebook.zip), [Mallcz](http://nlp.kiv.zcu.cz/data/research/sentiment/mallcz.zip)) 
   
   For the cross-lingual experiments you will also need to download the  [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/)   
   Copy and unzip the downloaded files into the corresponding folders `/data/polarity/[csfd/fb/mallcz/imdb]/original`    
   Run the split script `split_data.py`
    ```
   python3 split_data.py
   ```

   Download the Czech [fastText embeddings](https://fasttext.cc/docs/en/english-vectors.html) , copy and unzip into `/data/embeddings/cs` 
   
   Download the pre-trained [Czert models](https://drive.google.com/drive/folders/1o-PedUATiGyiSFG9gyj-xul30NXRfMFB?usp=sharing) , alternatively from [Czert github](https://github.com/kiv-air/Czert)
   and copy them to `/data/pre-trained` 

Fine-tuned Models:
--------
[PyTorch Fine-tuned models](https://drive.google.com/drive/folders/1vvbX_PmQvtw-2Vs-vgy7oKMFUeaRwH_A?usp=sharing)



Publication:
--------

If you use this software for academic research, please cite the following papers

```
@inproceedings{TODO
}


```

License:
--------
The code is licensed under the GNU LESSER GENERAL PUBLIC LICENSE License (see LICENSE.txt file), but the fine-tuned models fall under their source license, please see the cited papers for the 
corresponding licenses.

Please respect the licenses of the dependency packages.


Contact:
--------
pribanp@kiv.zcu.cz

[http://nlp.kiv.zcu.cz](http://nlp.kiv.zcu.cz)
