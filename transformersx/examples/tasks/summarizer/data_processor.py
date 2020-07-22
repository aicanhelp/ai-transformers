import itertools

from ai_harness.fileutils import FileLineReader
from ai_harness.xml2object import parse as parse_xml2obj
from transformersx.utils.text_utils import remove_emoji, simple_cut_word_sentences, get_ngrams, get_word_ngrams
import re


def read_lcsts_origin(file, on_doc):
    assert file and on_doc
    file_reader = FileLineReader()
    lines = []

    def handle_line(line: str, *args):
        if not line.startswith('</doc'):
            # The LSCTS is not well-form xml doc, remove the id property in <doc> element
            if line.startswith('<doc'): line = '<doc>'
            lines.append(line)
        else:
            on_doc(''.join(lines) + '\n')
            lines.clear()

    file_reader.pipe(handle_line).read(file)


def convert_lcsts_from_origin(origin_file, dest_file, handler=None):
    with open(dest_file, 'w') as f:
        read_lcsts_origin(origin_file, lambda doc: f.write(doc if not handler else handler(doc)))


def convert_lcsts_origin_to_flat_xml(origin_file, dest_file):
    convert_lcsts_from_origin(origin_file, dest_file)


def convert_xml_to_json(doc):
    obj = parse_xml2obj(doc)
    text = remove_emoji(obj.doc.short_text.cdata)
    summary = remove_emoji(obj.doc.sumary.cdata)
    return '{' + '"text":"{}","summary":"{}"'.format(text, summary) + '}\n'


def convert_lcsts_orgin_to_json(origin_file, dest_file):
    convert_lcsts_from_origin(origin_file, dest_file, convert_xml_to_json)


def make_abstract_labels(text, summary):
    return []


def convert_xml_to_abstract_example(doc):
    obj = parse_xml2obj(doc)
    text = simple_cut_word_sentences(remove_emoji(obj.doc.short_text.cdata))
    summary = make_abstract_labels(text, remove_emoji(obj.doc.sumary.cdata))
    return '{' + '"text":"{}","summary":"{}"'.format('\n'.join(text), ' '.join(summary)) + '}\n'


def build_lcsts_abstract_examples(origin_file, dest_file):
    convert_lcsts_from_origin(origin_file, dest_file, convert_xml_to_abstract_example)


