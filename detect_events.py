# coding: utf-8

# std
import timeit
import argparse

# mabed
from mabed.corpus import Corpus
from mabed.mabed import MABED
import mabed.utils as utils

__author__ = "Adrien Guille"
__email__ = "adrien.guille@univ-lyon2.fr"

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Perform mention-anomaly-based event detection (MABED)')
    p.add_argument('i', metavar='input', type=str, help='Input csv file')
    p.add_argument('k', metavar='top_k_events', type=int, help='Number of top events to detect')
    p.add_argument('--sw', metavar='stopwords', type=str, help='Stop-word list', default='stopwords/stopwords_hu.txt')
    p.add_argument('--sep', metavar='separator', type=str, help='CSV separator', default=',')
    p.add_argument('--o', metavar='output', type=str, help='Output pickle file', default=None)
    p.add_argument('--maf', metavar='min_absolute_frequency', type=int, help='Minimum absolute word frequency, default to 1', default=1)
    p.add_argument('--mrf', metavar='max_relative_frequency', type=float, help='Maximum absolute word frequency, default to 0.4', default=0.4)
    p.add_argument('--tsl', metavar='time_slice_length', type=int, help='Time-slice length, default to 1440 (minutes)', default=10080) # 1 day = 1440, 1 week = 10080
    p.add_argument('--p', metavar='p', type=int, help='Number of candidate words per event, default to 10', default=10)
    p.add_argument('--t', metavar='theta', type=float, help='Theta, default to 0.6', default=0.6)
    p.add_argument('--s', metavar='sigma', type=float, help='Sigma, default to 0.6', default=0.6)
    p.add_argument('--code', metavar='sigma', type=str, help='Encoding', default='utf-8')
    p.add_argument('--tt', metavar='sigma', type=str, help='The name of the header in the csv', default='fulltext')
    args = p.parse_args()
    print('Parameters:')
    print('   Corpus: %s\n   k: %d\n   Stop-words: %s\n   Min. abs. word frequency: %d\n   Max. rel. word frequency: %f' %
          (args.i, args.k, args.sw, args.maf, args.mrf))
    print('   p: %d\n   theta: %f\n   sigma: %f' % (args.p, args.t, args.s))

    print('Loading corpus...')
    start_time = timeit.default_timer()
    my_corpus = Corpus(args.i, args.sw, args.maf, args.mrf, args.sep, args.code)
    elapsed = timeit.default_timer() - start_time
    print('Corpus loaded in %f seconds.' % elapsed)

    time_slice_length = args.tsl
    print('Partitioning tweets into %d-minute time-slices...' % time_slice_length)
    start_time = timeit.default_timer()
    my_corpus.discretize(time_slice_length)
    elapsed = timeit.default_timer() - start_time
    print('Partitioning done in %f seconds.' % elapsed)

    print('Running MABED...')
    k = args.k
    p = args.p
    theta = args.t
    sigma = args.s
    start_time = timeit.default_timer()
    mabed = MABED(my_corpus)
    mabed.run(k=k, p=p, theta=theta, sigma=sigma)
    mabed.print_events()
    elapsed = timeit.default_timer() - start_time
    print('Event detection performed in %f seconds.' % elapsed)

    if args.o is not None:
        utils.save_events(mabed, args.o)
        print('Events saved in %s' % args.o)

    # output_filename = '../output/NER_Mabed_k20_tsl1.json'
    # mabed.save_events_json(output_filename)

