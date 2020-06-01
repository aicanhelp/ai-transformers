from tqdm.auto import tqdm
import random

from ..task_base import *


@configclass
class NewsDataArguments:
    data_dir: str = field('./dataset/news', 'input the data dir for processing')
    save_mid: bool = field(True, 'whether cache the middle data for check')
    context_min_len: int = field(128, 'context min length')
    sentence_min_len: int = field(10, 'sentence min length')
    positive_mode: int = field(0, 'positive mode')
    negative_mode: int = field(0, 'negative mode')
    bar_size: int = field(1000, 'the progress bar size')


class SentenceSplitter:
    def __init__(self, sentence_min_len=10):
        self._min_len = sentence_min_len

    def split(self, segment: str):
        t = segment.replace('\n', '')
        sentences = cut_sentences(t)
        length = len(sentences)
        sents = []
        i = 1
        l1 = len(sentences[i - 1])
        while i < length:
            l2 = len(sentences[i])
            if l1 > self._min_len or l1 != l2:
                sents.append(sentences[i - 1])
                if i == length - 1:
                    sents.append(sentences[i])
                i = i + 1
                l1 = l2
                continue
            start = i - 1
            while True:
                if i == length or sentences[i - 1][-1] != sentences[i][-1]:
                    break
                i = i + 1

            sents.append(''.join(sentences[start:i]))
            i = i + 1
            l1 = l2
        return sents


class SentencesSegment():
    def __init__(self, segment: str = None, sentence_min_len=10, exclude_empty_line=True):
        self._exclude_empty_line = exclude_empty_line
        self._sentences = self._make_sentences(segment, sentence_min_len)
        self._size = len(self._sentences)

    def _make_sentences(self, segment: str, sentence_min_len):
        if not segment: return []
        return SentenceSplitter(sentence_min_len).split(segment)

    def add_new_sentence(self, sentence: str):
        sentence = sentence.strip()
        if self._exclude_empty_line and not sentence:
            return
        self._sentences.append(sentence)
        self._size = self._size + 1

    def sentences(self, from_index, to_index):
        if to_index > self._size: to_index = self._size
        if from_index < 0: from_index = 0
        if from_index >= to_index: return None

        return ''.join(self._sentences[from_index:to_index])

    def first_sentences(self, len):
        if len < 1: return None
        len = self._size if len > self._size else len
        return self.sentences(0, len)

    def first_sentence(self):
        return self._sentences[0]

    def sentence(self, s_i):
        if s_i >= self._size: return None
        return self._sentences[s_i]

    def last_sentence(self):
        return self._sentences[-1]

    def size(self):
        return self._size

    def all_sentences(self):
        return self._sentences


class NewsExampleSegment(SentencesSegment):
    def __init__(self, segment: str, context_min_len=50, sentence_min_len=10):
        super().__init__(segment, sentence_min_len)
        self._context_min_len = context_min_len
        self.contexts = []
        self._make_contexts()

    def _make_contexts(self):
        total = self.size()
        if total < 1:
            return []

        length = len(self.first_sentence())
        start = 1
        while start < total and length < self._context_min_len:
            length = length + len(self.sentence(start))
            start = start + 1
        self.contexts.append((0, start))

        last_start = 0
        length = length - len(self.sentence(last_start))
        for end in range(start + 1, total):
            length = length + len(self.sentence(end))
            if length < self._context_min_len:
                continue

            last_start = last_start + 1
            self.contexts.append((last_start, end + 1))
            length = length - len(self.sentence(last_start))

    def context(self, c_start, c_end):
        return self.sentences(c_start, c_end)

    def last_context(self):
        return self.sentences(self.contexts[-1][0], self.contexts[-1][1])

    def example(self, c_start, c_end):
        return self.context(c_start, c_end), self.sentence(c_end)


class NewsExampleGenerator():
    def __init__(self, config: NewsDataArguments, type='train'):
        self._type = type
        self._config = config
        self.examples = []
        self._last_segment: NewsExampleSegment = None
        self._example_id = 0

    def add_line(self, new_segment_str: str):
        new_segment = NewsExampleSegment(new_segment_str,
                                         self._config.context_min_len,
                                         self._config.sentence_min_len)
        if not new_segment.size():
            return self

        if self._last_segment is not None:
            # In order to balance the number of classification,
            if eval('self._create_positive_examples_' + str(self._config.positive_mode) + '()'):
                eval('self._create_negative_examples_' + str(self._config.negative_mode) + '(new_segment)')

        self._last_segment = new_segment

        return self

    def _guid(self):
        self._example_id = self._example_id + 1
        return "%s-%s" % (self._type, self._example_id)

    def _create_positive_examples_0(self):
        if len(self._last_segment.contexts) < 2: return False
        e_index = random.choice(self._last_segment.contexts[:-1])
        guid = self._guid()
        text_a, text_b = self._last_segment.example(*e_index)
        self.examples.append(InputExample(guid=guid,
                                          text_a=text_a,
                                          text_b=text_b,
                                          label='0'))
        return True

    def _create_positive_examples_1(self):
        for e_index in self._last_segment.contexts[:-1]:
            guid = self._guid()
            text_a, text_b = self._last_segment.example(*e_index)
            self.examples.append(InputExample(guid=guid,
                                              text_a=text_a,
                                              text_b=text_b,
                                              label='0'))

    def _create_negative_examples_0(self, new_segment: NewsExampleSegment):
        guid = self._guid()
        text_a = self._last_segment.last_context()
        text_b = new_segment.sentence(0)
        self.examples.append(InputExample(guid=guid,
                                          text_a=text_a,
                                          text_b=text_b,
                                          label='1'))

    def _create_negative_examples_1(self, new_segment: NewsExampleSegment):
        guid = self._guid()
        for context in self._last_segment.contexts:
            text_a = self._last_segment.context(*context)
            for text_b in new_segment.all_sentences():
                self.examples.append(InputExample(guid=guid,
                                                  text_a=text_a,
                                                  text_b=text_b,
                                                  label='1'))


class FileNewsExampleProcessor:
    def __init__(self, file, config: NewsDataArguments, type='train'):
        self._file = file
        self._config = config
        self._type = type
        self._example_generator = NewsExampleGenerator(config, type)

    def _make_examples(self):
        def handle_line(line, previous):
            self._example_generator.add_line(line)

        return handle_line

    def _save_middle_data(self):
        log.info("Save the middle data: ")
        with open(join_path(self._config.data_dir, self._type + "_middle_examples.txt"), 'w') as f:
            for example in tqdm(self._example_generator.examples, disable=(self._config.bar_size <= 0)):
                f.write(example.text_a + '\n')
                f.write(example.text_b + '\n')
                f.write(example.label + '\n\n')

    def get_examples(self):
        FileLineReader(self._config.bar_size).pipe(self._make_examples()).read(self._file)
        if self._config.save_mid:
            self._save_middle_data()
        return self._example_generator.examples


class NewsDataProcessor(DataProcessor):
    def __init__(self, config: NewsDataArguments):
        self._config = config
        if self._config is None:
            self._config = parse_tasks_args(NewsDataArguments)

    def _get_example(self, file_name, type):
        return FileNewsExampleProcessor(join_path(self._config.data_dir, file_name), self._config, type).get_examples()

    def get_train_examples(self):
        return self._get_example('train.txt', 'train')

    def get_dev_examples(self):
        return self._get_example('dev.txt', 'dev')

    def get_labels(self):
        return ['0', '1']

    def data_dir(self):
        return self._config.data_dir


class PredictDataProcessor(DataProcessor):
    def __init__(self, segment: NewsExampleSegment):
        self._segment = segment

    def get_train_examples(self):
        return None

    def get_dev_examples(self):
        return self._create_example_from_article()

    def _create_example_from_article(self):
        examples = []
        for i, e_index in enumerate(self._segment.contexts[:-1]):
            text_a, text_b = self._segment.example(*e_index)
            examples.append(InputExample(guid=str(i),
                                         text_a=text_a,
                                         text_b=text_b))
        return examples

    def get_labels(self):
        return ['0', '1']

    def data_dir(self):
        return None
