from transformersx.utils.text_utils import make_sentences_max_length, cut_sentences, simple_cut_word_sentences, \
    rouge_clean

content = ['斯坦议会上院议长纳扎尔巴耶娃和,',
           '下面一场尼格马图林邀请全国人大常委会委员长栗战书，21号到25号对哈萨克斯坦进行正式友好访问，再努尔苏丹会见首任总统纳扎尔巴耶夫和总统托卡耶夫，与娜扎尔巴耶娃尼格马图林分别举行会谈，与总理马明共同出席相关活动。',
           '中哈',
           '关系实现',
           '中哈关系实现历史性跨越，成为国家间关系的典范。'
           ]
content2 = [
    '其实热点多方关注，在今天的新闻最关注之中，我们来关注记者张美所采写的报道，守护人与自然的和谐巍峨的大秦岭是我国南北气候的分界线，生物的多样性使其成为了地球自然资源保护西成高铁是我国首条穿越秦岭山脉的高铁，途经全国最大的连线自然保护区，当您乘坐西成高铁驰骋在秦岭山脉之间时，或许能够看见美丽的猪还飞过头顶，但是很少有人知道，这幅人与自然和谐共存的画面背后，是一支由动物保护专家组成的科研团队，日复一日年复一年的付出！']


def test_make_sentences_max_length():
    sentences = make_sentences_max_length(content2)
    print(sentences)


def test_simple_cut_sentencse():
    sentences = simple_cut_word_sentences(content2[0])
    print(sentences)


def test_rouge_clean():
    print(rouge_clean(content2[0]))
