# Tensorflow-Text-Generator
This code uses LSTM neural networks in tensorflow to generate text.
## Usage
For training the model you can set the textfile you want to use by using command line.

For training the network run:

```
$ python text_generation_tensorflow.py --input_file=data/shakespeare.txt --ckpt_file="saved/model.ckpt" --mode=train
```
For generating the text run:

```
$ python text_generation_tensorflow.py --input_file=data/shakespeare.txt --ckpt_file="saved/model.ckpt" --test_prefix="The " --mode=talk
```
