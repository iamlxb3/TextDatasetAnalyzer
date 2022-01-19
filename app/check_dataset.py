import sys
import ipdb
import math
import ntpath
import random
import numpy as np
import pandas as pd

sys.path.append('..')
import argparse
from tabulate import tabulate
from core.text_analyzer import TextAnalyzer

__VALID_ASPECT_CHOICES = ('distinct', 'basic', 'pos', 'dep', 'ner', 'stopword', 'zipflaw', 'concreteness')


# python check_dataset.py --path ../data/cn_example --n_grams 1 2 3 --aspects distinct basic --lang cn
# python check_dataset.py --path ../data/cn_example --n_grams 1 2 3 --aspects pos --lang cn --sample_N 10
# python check_dataset.py --path ../data/cn_example --aspects zipflaw --lang cn --sample_N 10 --n_grams 1 2
# python check_dataset.py --path ../data/cn_example --aspects concreteness --lang cn --sample_N 10
# python check_dataset.py --path ../data/cn_example --aspects all --lang cn --n_grams 1 2 --sample_N 10
# python check_dataset.py --path ../data/en_example1 --aspects all --lang en --n_grams 1 2
# python check_dataset.py --path ../data/en_example2 --aspects all --lang en --n_grams 1 2

def args_parse():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--aspects', type=str, nargs='+', choices=list(__VALID_ASPECT_CHOICES) + ['all'],
                        required=True)
    parser.add_argument('--n_grams', nargs='+', type=int)
    parser.add_argument('--lang', type=str, choices=['en', 'cn'], required=True)
    parser.add_argument('--sample_N', type=int)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--cache_spacy_result', action="store_true",
                        help='whether to cache spacy result by saving to disk', default=True)
    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    path = args.path
    n_grams = args.n_grams
    lang = args.lang
    aspects = args.aspects
    sample_N = args.sample_N
    dataset = args.dataset
    cache_spacy_result = args.cache_spacy_result

    if dataset is None:
        dataset = ntpath.basename(path)
        if '.' in dataset:
            dataset = dataset.split('.')[0]

    if len(aspects) == 1 and aspects[0] == 'all':
        aspects = __VALID_ASPECT_CHOICES

    if 'zipflaw' in aspects:
        assert n_grams is not None

    random.seed(0)

    # init text_analyzer
    if 'pos' in aspects or 'ner' in aspects or 'dep' in aspects or 'concreteness' in aspects:
        text_analyzer = TextAnalyzer(language=lang, load_spacy_model=True)
    else:
        text_analyzer = TextAnalyzer(language=lang)

    # read dataset
    text = text_analyzer.read_text_from_file(path)
    if sample_N is not None:
        text = random.sample(text, sample_N)
    print("=" * 78)
    print(f"Read dataset from {path}, language: {lang}, size: {len(text)}")
    print("=" * 78)

    for aspect in aspects:
        if aspect == 'distinct':
            # tokens seperated by whitespace
            ngram_df = text_analyzer.compute_ngram_distinct(text, n_grams=tuple(n_grams))
            print(tabulate(ngram_df, headers='keys', tablefmt='psql'))

        if aspect == 'basic':
            text_analyzer.check_basic(text)

        if aspect in {'pos', 'dep', 'ner'}:
            parse_result = text_analyzer.load_parsed_texts_by_spacy(text, aspect, dump_res=cache_spacy_result)
            print("-" * 78)
            print(f"Spacy {aspect} result")
            print("-" * 78)
            print(tabulate(parse_result))

        if aspect == 'stopword':
            stopword_sen_ratio, stopwords = text_analyzer.analyse_stopwords(text)
            sorted_stopwords = sorted(stopwords.items(), key=lambda x: x[1], reverse=True)[:10]
            top_stopwords = pd.DataFrame(sorted_stopwords)
            print("-" * 78)
            print(f"Top 10 stopwords")
            print("-" * 78)
            print(tabulate(top_stopwords))
            print(f"The average percentage of stopwords in a sentence: {np.average(stopword_sen_ratio)}")
            stopword_sen_ratio_df = pd.DataFrame({'value': stopword_sen_ratio,
                                                  'language': [lang for _ in stopword_sen_ratio],
                                                  'dataset': [dataset for _ in stopword_sen_ratio]})
            stopword_dis_save_path = f'../results/{dataset}_{lang}_stopwords.csv'
            stopword_sen_ratio_df.to_csv(stopword_dis_save_path, index=False)
            print(f"Save stopword distribution df to {stopword_dis_save_path}")

        if aspect == 'zipflaw':
            zipflaw_df = {'rank': [], 'freq': [], 'token': [], 'dataset': [], 'n_gram': []}
            for n_gram in n_grams:
                n_gram_freq, n_gram_idf = text_analyzer.compute_ngram(text, n_gram=n_gram)
                n_gram_freq = sorted(n_gram_freq.items(), key=lambda x: x[1], reverse=True)
                for i, (token, freq) in enumerate(n_gram_freq):
                    zipflaw_df['rank'].append(i)
                    zipflaw_df['freq'].append(math.log(freq))
                    zipflaw_df['token'].append(token)
                    zipflaw_df['dataset'].append(dataset)
                    zipflaw_df['n_gram'].append(n_gram)
            zipflaw_df = pd.DataFrame(zipflaw_df)
            zipflaw_save_path = f'../results/{dataset}_{lang}_zipflaw.csv'
            zipflaw_df.to_csv(zipflaw_save_path, index=False)
            print(f"Save zipflaw distribution df to {zipflaw_save_path}")

        if aspect == 'concreteness':
            concreteness_df = {'value': [], 'language': [], 'dataset': [], 'pos': []}
            concreteness = text_analyzer.analyse_concreteness(text, lang)
            for pos, value in concreteness.items():
                concreteness_df['value'].extend(value)
                concreteness_df['language'].extend([lang for _ in value])
                concreteness_df['dataset'].extend([dataset for _ in value])
                concreteness_df['pos'].extend([pos for _ in value])
            concreteness_df = pd.DataFrame(concreteness_df)
            concreteness_save_path = f'../results/{dataset}_{lang}_concreteness.csv'
            concreteness_df.to_csv(concreteness_save_path, index=False)
            print(f"Save concreteness distribution df to {concreteness_save_path}")


if __name__ == '__main__':
    main()
