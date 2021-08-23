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
```
--dataset_name fb --enable_wandb --model_name lstm-base --embeddings_file cc.cs.300.vec --embeddings_size 300 --max_words 300000 --trainable_word_embeddings --tokenizer toktok --use_stemmer --use_data_cleaner --max_seq_len 512 --num_repeat 1 --use_attention --attention_type multiplicative --rnn_cells 256 --epoch_count 5 --optimizer AdamW --batch_size 256 --lr 0.0005 --lr_scheduler_name cosine --warm_up_steps 0.1 --use_only_train_data --weight_decay 0.0001
--dataset_name fb --enable_wandb --model_name lstm-base --embeddings_file cc.cs.300.vec --embeddings_size 300 --max_words 300000 --trainable_word_embeddings --tokenizer toktok --use_stemmer --use_data_cleaner --max_seq_len 512 --num_repeat 1 --use_attention --attention_type multiplicative --rnn_cells 256 --epoch_count 5 --optimizer AdamW --batch_size 256 --lr 0.0005 --lr_scheduler_name cosine --warm_up_steps 0.1 --use_only_train_data --weight_decay 0.0001 --binary


--dataset_name csfd --enable_wandb --model_name lstm-base --embeddings_file cc.cs.300.vec --embeddings_size 300 --max_words 300000 --trainable_word_embeddings --tokenizer toktok --use_stemmer --use_data_cleaner --max_seq_len 512 --num_repeat 1 --use_attention --attention_type multiplicative --rnn_cells 128 --epoch_count 2 --optimizer AdamW --batch_size 128 --lr 0.0005 --lr_scheduler_name cosine --warm_up_steps 0.1 --use_only_train_data
--dataset_name csfd --enable_wandb --model_name lstm-base --embeddings_file cc.cs.300.vec --embeddings_size 300 --max_words 300000 --trainable_word_embeddings --tokenizer toktok --use_stemmer --use_data_cleaner --max_seq_len 512 --num_repeat 1 --use_attention --attention_type multiplicative --rnn_cells 128 --epoch_count 2 --optimizer AdamW --batch_size 128 --lr 0.0005 --lr_scheduler_name cosine --warm_up_steps 0.1 --use_only_train_data --binary

--dataset_name mallcz --enable_wandb --model_name lstm-base --embeddings_file cc.cs.300.vec --embeddings_size 300 --max_words 300000 --trainable_word_embeddings --tokenizer toktok --use_stemmer --use_data_cleaner --max_seq_len 512 --num_repeat 1 --use_attention --attention_type multiplicative --rnn_cells 128 --epoch_count 10 --optimizer AdamW --batch_size 128 --lr 0.0005 --lr_scheduler_name cosine --warm_up_steps 0.1 --use_only_train_data
--dataset_name mallcz --enable_wandb --model_name lstm-base --embeddings_file cc.cs.300.vec --embeddings_size 300 --max_words 300000 --trainable_word_embeddings --tokenizer toktok --use_stemmer --use_data_cleaner --max_seq_len 512 --num_repeat 1 --use_attention --attention_type multiplicative --rnn_cells 128 --epoch_count 2 --optimizer AdamW --batch_size 128 --lr 0.0005 --lr_scheduler_name cosine --warm_up_steps 0.1 --use_only_train_data --binary


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
The code is licensed under the MIT License (see LICENSE.txt file), but the fine-tuned models fall under their source license, please see the cited papers for the 
corresponding licenses.


Contact:
--------
pribanp@kiv.zcu.cz

[http://nlp.kiv.zcu.cz](http://nlp.kiv.zcu.cz)
