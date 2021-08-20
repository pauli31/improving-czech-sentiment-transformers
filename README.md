# Improving Czech Sentiment with Transformers
Repository for paper:
##Are the Multilingual Models Better? Improving Czech Sentiment with Transformers

If you use this software for academic research, [please cite the paper](#publication)


Usage:
--------
FB Baseline LR
```
python3 baseline.py --dataset_name fb --use_train_test --word_ngram_vectorizer cv --char_ngram_vectorizer cv
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
3 Classes

Model | CSFD | FB | Mallcz  
--- | --- | --- | --- 
Czert-A | [link]() |  |  
Czert-B |  |  |  
mBERT |  |  |  
SlavicBERT |  |  |  
RandomALBERT |  |  |  
XLM-R-Base |  |  |  
XLM-R-Large |  |  |  
XLM |  |  |  

2 - classes

Model | CSFD | FB | Mallcz  
--- | --- | --- | --- 
Czert-A |  |  |  
Czert-B |  |  |  
mBERT |  |  |  
SlavicBERT |  |  |  
RandomALBERT |  |  |  
XLM-R-Base |  |  |  
XLM-R-Large |  |  |  
XLM |  |  |  

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
