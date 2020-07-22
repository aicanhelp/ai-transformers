import itertools

from transformersx.utils.text_utils import get_word_ngrams, rouge_clean


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count, evaluated_count = len(reference_ngrams), len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    precision = 0.0 if evaluated_count == 0 else overlapping_count / evaluated_count
    recall = 0.0 if reference_count == 0 else overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def combination_selection(doc_sent_list, abstract_sent_list, summary_size):
    max_rouge, max_idx = 0.0, (0, 0)
    abstract = sum(abstract_sent_list, [])
    abstract = rouge_clean(' '.join(abstract)).split()
    sents = [rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = get_word_ngrams(1, [abstract])
    evaluated_2grams = [get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = get_word_ngrams(2, [abstract])

    impossible_sents = []
    for s in range(summary_size + 1):
        combinations = itertools.combinations([i for i in range(len(sents)) if i not in impossible_sents], s + 1)
        for c in combinations:
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']

            rouge_score = rouge_1 + rouge_2
            if (s == 0 and rouge_score == 0):
                impossible_sents.append(c[0])
            if rouge_score > max_rouge:
                max_idx = c
                max_rouge = rouge_score
    return sorted(list(max_idx))


def greedy_selection(doc_sent_list, abstract_sent_list, summary_size):
    max_rouge = 0.0
    abstract = sum(abstract_sent_list, [])
    abstract = rouge_clean(' '.join(abstract)).split()
    sents = [rouge_clean(' '.join(s)).split() for s in doc_sent_list]
    evaluated_1grams = [get_word_ngrams(1, [sent]) for sent in sents]
    reference_1grams = get_word_ngrams(1, [abstract])
    evaluated_2grams = [get_word_ngrams(2, [sent]) for sent in sents]
    reference_2grams = get_word_ngrams(2, [abstract])

    selected = []
    for s in range(summary_size):
        cur_max_rouge = max_rouge
        cur_id = -1
        for i in range(len(sents)):
            if (i in selected):
                continue
            c = selected + [i]
            candidates_1 = [evaluated_1grams[idx] for idx in c]
            candidates_1 = set.union(*map(set, candidates_1))
            candidates_2 = [evaluated_2grams[idx] for idx in c]
            candidates_2 = set.union(*map(set, candidates_2))
            rouge_1 = cal_rouge(candidates_1, reference_1grams)['f']
            rouge_2 = cal_rouge(candidates_2, reference_2grams)['f']
            rouge_score = rouge_1 + rouge_2
            if rouge_score > cur_max_rouge:
                cur_max_rouge = rouge_score
                cur_id = i
        if (cur_id == -1):
            return selected
        selected.append(cur_id)
        max_rouge = cur_max_rouge

    return sorted(selected)
