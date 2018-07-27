# Tensorflow-Text-Generator
# Usage
To train the model you can set the textfile you want to use to train the model by using command line:

For training the network run:

```
$ python text_generation_tensorflow.py --input_file=data/shakespeare.txt --ckpt_file="saved/model.ckpt" --mode=train
```
For generating the text run:

```
$ python text_generation_tensorflow.py --input_file=data/shakespeare.txt --ckpt_file="saved/model.ckpt" --test_prefix="The " --mode=talk
```
