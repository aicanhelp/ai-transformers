# ai-transformers

A trick for downloading pretrained models from huggingface easily, especially in China.  


Because it's hard to download the pretrained models from huggingface, especially in China, here tries to use a trick to 
solve this problem.  

For personal, Aliyun is free to build the docker image, typically, the image building can use the overseas machine 
to build the docker image. So when the image is built by the overseas machine, it can download the pretrained models from
huggingface fastly.  

After the image is built, the image can be pull from Aliyun fastly. And then the pretrained models can be take from the docker image.

