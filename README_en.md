# Transformersx

[🤗 Transformers](https://github.com/huggingface/transformers) is a great project for the Transformer architecture for NLP and NLG.  

/**🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides state-of-the-art general-purpose 
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet, T5, CTRL...) for Natural Language Understanding (NLU) and 
Natural Language Generation (NLG) with over thousands of pretrained models in 100+ languages and deep interoperability 
between PyTorch & TensorFlow 2.0.**/

The purpose of this project is to add more tools to easy to use the hugging/transformers library for NLP.
And NLP task examples was refactored or added as well.

/**BTW, **/
Because it's hard to download the pretrained models from huggingface, especially in China, here tries to use a trick to 
solve this problem.  

For personal, Aliyun is free to build the docker image, typically, the image building can use the overseas machine 
to build the docker image. So when the image is built by the overseas machine, it can download the pretrained models from
huggingface fastly.  

After the image is built, the image can be pulled from Aliyun fastly. And then the pretrained models can be take from the docker image.



#### TODO:  
- (1) Build a management tools for the models of transformers to categorize the models, 
    and the features also should be including training, finetuning and predict
- (2) Use the Streamlit to build the UI for the management tools



