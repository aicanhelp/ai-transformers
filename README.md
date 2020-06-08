# Transformersx

##介绍
[🤗 Transformers](https://github.com/huggingface/transformers) 是一个非常好用的专门针对基于Pytorch的Transformer相关NLP深度学习模型的工具库。
它管理和归类了当前几乎所有的最好的基于的Transformer的自然语言模型以及公开的预训练模型，并都转换成了Pytorch。
（1）使用它你可以很方便的做Bert/Albert/GPT2/XLNET等当前最好的自然语言预训练模型训练以及下游任务的模型开发和训练。
（2）使用Pytorch实现的模型有更清晰的代码机构，对学习者来说，学习和理解这些自然语言更为容易。
（3）提供了一个地方收集和存放公开的自然语言预训练模型，供研究人员使用。研究人员也可以把自己愿意公开的预训练模型放到Transformer上供别人研究使用。

但是Transformers也有一些问题：
（1）根据当前的Transformers的实现看，所有的公开的Transformer的预训练模型都是存放在AWS的S3上。对国外的研究人员那没有什么问题。中国的研究人员需要下载这些模型
就有点费周折了。当前Transformers的模型下载方式有两种：（1）通过代码调用指定模型时，自动去相应的S3上下载。（2）直接到Transformers的网上流量相应的模型文件通过浏览器下载。
反正不管那种方式，想顺利的下载下来，自己想办法吧～。不多说。
（2）尽管Transformers已经提供了一种比较方便的方式来使用各种Transformer相关模型了。但是还是不够好。首先，从设计上，各种模型的实现很不错，但是因为模型的实现与模型的存储和下载深度绑定。
这个设计应该是有问题的。从职责上说，模型的实现和模型的存储下载应该分离。
（3）Transformers增加了一个Trainer以方便研究人员训练Transformer模型使用。同样，这个Trainer的设计和实现水平跟模型的设计和实现也一样有不少的差距。

## 目的
本项目的目的是想针对Transformers的一些问题，对Transformers做进一步的扩展，让研究人员使用Transformers更方便。当然也没有解决上面提到的所有问题。
### （1）首先，针对下载这个问题。  
本项目的解决方法是，在docker目录中提供了几种用于Transformers相关模型的训练和运行环境的Docker定义，
利用阿里云的Docker海外构建机器，在构建Docker是顺便把指定的预训练模型下载下来。当前主要是中文语言模型，包括bert、albert、robert、electra。可以直接从阿里云镜像库获取。
相关模型放在镜像的/app/models目录下面。
```
docker pull registry.cn-beijing.aliyuncs.com/modoso/transformersx-bert
docker pull registry.cn-beijing.aliyuncs.com/modoso/transformersx-albert
docker pull registry.cn-beijing.aliyuncs.com/modoso/transformersx-robert
docker pull registry.cn-beijing.aliyuncs.com/modoso/transformersx-electra
```
最好使用脚本download-models.sh 下载并从镜像中把模型copy出来的脚本。
```
sh download-models.sh [指定模型存放目录]
```
### （2）为了更方便的使用，(当然，你得先参照上面的模型下载方式下载docker镜像或者模型)
首先，你可以简单的像examples.task.sentiment.sentiment_task那样实现情感识别，只需要实现一个DataProcessor和一个Task

```python
from ai_transformersx import DataProcessor,DataArguments,join_path,InputExample,log,TaskArguments
from ai_transformersx.examples import ExampleTaskBase
import pandas as pd
class SentimentDataProcessor(DataProcessor):
    def __init__(self, config: DataArguments):
        self._config = config

    def _get_example(self, file_name, type):
        pd_all = pd.read_csv(join_path(self._config.data_dir, file_name))

        log.info("Read data from {}, length={}".format(join_path(self._config.data_dir, file_name), len(pd_all)))
        examples = []
        for i, d in enumerate(pd_all.values):
            examples.append(InputExample(guid=type + '_' + str(i),
                                         text_a=d[1],
                                         label=str(d[0])))

        return examples

    def get_train_examples(self):
        return self._get_example('train.csv', 'train')

    def get_dev_examples(self):
        return self._get_example('dev.csv', 'dev')

    def get_labels(self):
        return ['0', '1', '2', '3']

    def data_dir(self):
        return self._config.data_dir


class SentimentTask(ExampleTaskBase):
    def __init__(self, taskArgs: TaskArguments = None):
        super().__init__('sentiment', taskArgs)
        self.task_args.model_args.num_labels = 4

    def _data_processor(self):
        return SentimentDataProcessor(self.task_args.data_args)
```
然后，像examples.task.main那样实现启动方法：
```python
from ai_transformersx.examples import ExampleManagement
from ai_transformersx.examples.tasks import SentimentTask
task_manager = ExampleManagement()
task_manager.register_tasks([
    ('sentiment', SentimentTask)
])

if __name__ == "__main__":
    task_manager.start_example_task()

```
接着，你就可以训练你的情感识别的模型了。你应该在下载的镜像模型中进行。
- 查看所有sentiment任务参数
```
    python main.py sentiment -h
```
- 训练模型（可以参考上面的帮助列表设置相关的参数）
```
    python main.py sentiment
```

###  (3) 常见的中文的自然语言任务的例子

## TODO:  
- 使用 pytorchlightning和fastai来实现trainer
- 增加更多的自然语言任务的例子



