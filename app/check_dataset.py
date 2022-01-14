import sys
import ipdb

sys.path.append('..')
import argparse
from tabulate import tabulate
from core.text_analyzer import TextAnalyzer


# python check_dataset.py --path ../data/cn_dario_train --n_grams 1 2 3 --by distinct --lang cn

def args_parse():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--by', type=str, choices=['distinct', 'all'], required=True)
    parser.add_argument('--n_grams', nargs='+', type=int)
    parser.add_argument('--lang', type=str, choices=['en', 'cn'], required=True)
    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    path = args.path
    n_grams = args.n_grams
    lang = args.lang
    by = args.by

    # init text_analyer
    text_analyer = TextAnalyzer(language=lang)

    # read dataset
    text = text_analyer.read_text_from_file(path)
    print("=" * 78)
    print(f"Read dataset from {path}, language: {lang}, size: {len(text)}")
    print("=" * 78)

    if by == 'distinct':
        # tokens seperated by whitespace
        ngram_df = text_analyer.compute_ngram_distinct(text, n_grams=tuple(n_grams))
        print(tabulate(ngram_df, headers='keys', tablefmt='psql'))


if __name__ == '__main__':
    main()
