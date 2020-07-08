import re


def cut_sentences(para, language="cn", drop_empty_line=True, strip=True, deduplicate=False, max_len=64, min_len=10):
    '''cut_sentences

    :param para: 输入文本
    :param drop_empty_line: 是否丢弃空行
    :param strip: 是否对每一句话做一次strip
    :param deduplicate: 是否对连续标点去重，帮助对连续标点结尾的句子分句
    :return: sentences: list of str
    '''
    if deduplicate:
        para = re.sub(r"([。！？\!\?])\1+", r"\1", para)

    if language == 'en':
        from nltk import sent_tokenize
        sents = sent_tokenize(para)
        if strip:
            sents = [x.strip() for x in sents]
        if drop_empty_line:
            sents = [x for x in sents if len(x.strip()) > 0]
        return sents
    else:
        para = re.sub('([。！？\?!])([^”’])', r"\1\n\2", para)  # 单字符断句符
        para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
        para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?!][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        sentences = para.split("\n")

        if strip:
            sentences = [sent.strip() for sent in sentences]
        if drop_empty_line:
            sentences = [sent for sent in sentences if len(sent.strip()) > 0]

    return make_sentences_max_length(sentences, max_len, min_len)


def make_sentences_max_length(sentences, max_len=64, min_len=10):
    if max_len < 1 and min_len < 1:
        return sentences

    new_sentences = []
    for s in sentences:
        # 拆分长句子，应该进一采用逗号分割
        if max_len > 0 and len(s) > max_len + min_len:
            sub_sentences = s.split('，')
            new_sentences.append(sub_sentences[0])
            if len(sub_sentences) == 1: continue

            new_sentences[-1] = new_sentences[-1] + '，'
            for sub_s in sub_sentences[1:-1]:
                if len(new_sentences[-1]) < max_len:
                    new_sentences[-1] = new_sentences[-1] + sub_s + '，'
                else:
                    new_sentences.append(sub_s + '，')

            if len(sub_sentences[-1]) < (max_len + min_len) / 2:
                new_sentences[-1] = new_sentences[-1] + sub_sentences[-1]
            else:
                new_sentences.append(sub_sentences[-1])

            continue

        if len(s) < min_len:
            new_sentences[-1] = new_sentences[-1] + s
        else:
            new_sentences.append(s)

    return new_sentences
