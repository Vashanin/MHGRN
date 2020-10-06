import os
import argparse
from multiprocessing import cpu_count
from utils.convert_csqa import convert_to_entailment
from utils.convert_scitail import convert_to_scitail_statement
from utils.convert_phys import convert_to_phys_statement
from utils.convert_socialiqa import convert_to_socialiqa_statement
from utils.convert_obqa import convert_to_obqa_statement
from utils.tokenization_utils import tokenize_statement_file, make_word_vocab
from utils.conceptnet import extract_english, construct_graph
from utils.embedding import glove2npy, load_pretrained_embeddings
from utils.grounding import create_matcher_patterns, ground
from utils.paths import find_paths, score_paths, prune_paths, find_relational_paths_from_paths, generate_path_and_graph_from_adj
from utils.graph import generate_graph, generate_adj_data_from_grounded_concepts, coo_to_normalized
from utils.triples import generate_triples_from_adj

input_paths = {
    'dataset': {
        'train': './data/{dataset}/train_rand_split.jsonl',
        'dev': './data/{dataset}/dev_rand_split.jsonl',
        'test': './data/{dataset}/test_rand_split_no_answers.jsonl',
    },
    'graph': {
        'csv': './data/{graph}/assertions-5.6.0.csv',
        'ent': './data/{graph}/transe/transe.sgd.ent.npy',
        'rel': './data/{graph}/transe/transe.sgd.rel.npy'
    },
    'glove': {
        'txt': './data/glove/glove.6B.300d.txt',
    },
    'numberbatch': {
        'txt': './data/transe/numberbatch-en-19.08.txt',
    }
}

output_paths = {
    'graph': {
        'csv': './data/{graph}/{graph}.en.csv',
        'vocab': './data/{graph}/concept.txt',
        'relations': './data/{graph}/relations.txt',
        'patterns': './data/{graph}/matcher_patterns.json',
        'unpruned-graph': './data/{graph}/{graph}.en.unpruned.graph',
        'pruned-graph': './data/{graph}/{graph}.en.pruned.graph',
    },
    'glove': {
        'npy': './data/glove/glove.6B.300d.npy',
        'vocab': './data/glove/glove.vocab',
    },
    'numberbatch': {
        'npy': './data/transe/nb.npy',
        'vocab': './data/transe/nb.vocab',
        'concept_npy': './data/transe/concept.nb.npy'
    },
    'dataset': {
        'statement': {
            'train': './data/{dataset}/statement/train.statement.jsonl',
            'dev': './data/{dataset}/statement/dev.statement.jsonl',
            'test': './data/{dataset}/statement/test.statement.jsonl',
            'vocab': './data/{dataset}/statement/vocab.json',
        },
        'statement-with-ans-pos': {
            'train': './data/{dataset}/statement/train.statement-with-ans-pos.jsonl',
            'dev': './data/{dataset}/statement/dev.statement-with-ans-pos.jsonl',
            'test': './data/{dataset}/statement/test.statement-with-ans-pos.jsonl',
        },
        'tokenized': {
            'train': './data/{dataset}/tokenized/train.tokenized.txt',
            'dev': './data/{dataset}/tokenized/dev.tokenized.txt',
            'test': './data/{dataset}/tokenized/test.tokenized.txt',
        },
        'grounded': {
            'train': './data/{dataset}/grounded/train.grounded.jsonl',
            'dev': './data/{dataset}/grounded/dev.grounded.jsonl',
            'test': './data/{dataset}/grounded/test.grounded.jsonl',
        },
        'paths': {
            'raw-train': './data/{dataset}/paths/train.paths.raw.jsonl',
            'raw-dev': './data/{dataset}/paths/dev.paths.raw.jsonl',
            'raw-test': './data/{dataset}/paths/test.paths.raw.jsonl',
            'scores-train': './data/{dataset}/paths/train.paths.scores.jsonl',
            'scores-dev': './data/{dataset}/paths/dev.paths.scores.jsonl',
            'scores-test': './data/{dataset}/paths/test.paths.scores.jsonl',
            'pruned-train': './data/{dataset}/paths/train.paths.pruned.jsonl',
            'pruned-dev': './data/{dataset}/paths/dev.paths.pruned.jsonl',
            'pruned-test': './data/{dataset}/paths/test.paths.pruned.jsonl',
            'adj-train': './data/{dataset}/paths/train.paths.adj.jsonl',
            'adj-dev': './data/{dataset}/paths/dev.paths.adj.jsonl',
            'adj-test': './data/{dataset}/paths/test.paths.adj.jsonl',
        },
        'graph': {
            'train': './data/{dataset}/graph/train.graph.jsonl',
            'dev': './data/{dataset}/graph/dev.graph.jsonl',
            'test': './data/{dataset}/graph/test.graph.jsonl',
            'adj-train': './data/{dataset}/graph/train.graph.adj.pk',
            'adj-dev': './data/{dataset}/graph/dev.graph.adj.pk',
            'adj-test': './data/{dataset}/graph/test.graph.adj.pk',
            'nxg-from-adj-train': './data/{dataset}/graph/train.graph.adj.jsonl',
            'nxg-from-adj-dev': './data/{dataset}/graph/dev.graph.adj.jsonl',
            'nxg-from-adj-test': './data/{dataset}/graph/test.graph.adj.jsonl',
        },
        'triple': {
            'train': './data/{dataset}/triples/train.triples.pk',
            'dev': './data/{dataset}/triples/dev.triples.pk',
            'test': './data/{dataset}/triples/test.triples.pk',
        }
    }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', choices=['cpnet', 'quasimodo'], help='graph to use for augmentation')
    parser.add_argument('--dataset', choices=['csqa', 'mnli'], help='dataset to be processed')
    parser.add_argument('--run', default=['common', 'dataset'], nargs='+')
    parser.add_argument('--path_prune_threshold', type=float, default=0.12, help='threshold for pruning paths')
    parser.add_argument('--max_node_num', type=int, default=200, help='maximum number of nodes per graph')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'common': [
            # {'func': glove2npy,
            #  'args': (input_paths['glove']['txt'],
            #           output_paths['glove']['npy'],
            #           output_paths['glove']['vocab'])},
            # {'func': glove2npy,
            #  'args': (input_paths['numberbatch']['txt'],
            #           output_paths['numberbatch']['npy'],
            #           output_paths['numberbatch']['vocab'], True)},
            # {'func': extract_english,
            #  'args': (input_paths['graph']['csv'].format(graph=args.graph),
            #           output_paths['graph']['csv'].format(graph=args.graph),
            #           output_paths['graph']['vocab'].format(graph=args.graph))},
            # {'func': load_pretrained_embeddings,
            #  'args': (output_paths['numberbatch']['npy'],
            #           output_paths['numberbatch']['vocab'],
            #           output_paths['graph']['vocab'].format(graph=args.graph),
            #           False,
            #           output_paths['numberbatch']['concept_npy'])},
            {'func': construct_graph,
             'args': (output_paths['graph']['csv'].format(graph=args.graph),
                      output_paths['graph']['vocab'].format(graph=args.graph),
                      output_paths['graph']['relations'].format(graph=args.graph),
                      output_paths['graph']['unpruned-graph'].format(graph=args.graph),
                      False)},
            {'func': construct_graph,
             'args': (output_paths['graph']['csv'].format(graph=args.graph),
                      output_paths['graph']['vocab'].format(graph=args.graph),
                      output_paths['graph']['relations'].format(graph=args.graph),
                      output_paths['graph']['pruned-graph'].format(graph=args.graph),
                      True)},
            {'func': create_matcher_patterns,
             'args': (output_paths['graph']['vocab'].format(graph=args.graph),
                      output_paths['graph']['patterns'].format(graph=args.graph))},
        ],
        'dataset': [
            {'func': convert_to_entailment,
             'args': (input_paths['dataset']['train'].format(dataset=args.dataset),
                      output_paths['dataset']['statement']['train'].format(dataset=args.dataset))},
            {'func': convert_to_entailment,
             'args': (input_paths['dataset']['dev'].format(dataset=args.dataset),
                      output_paths['dataset']['statement']['dev'].format(dataset=args.dataset))},
            {'func': convert_to_entailment,
             'args': (input_paths['dataset']['test'].format(dataset=args.dataset),
                      output_paths['dataset']['statement']['test'].format(dataset=args.dataset))},

            # {'func': tokenize_statement_file,
            #  'args': (output_paths[args.dataset]['statement']['train'],
            #           output_paths[args.dataset]['tokenized']['train'])},
            # {'func': tokenize_statement_file,
            #  'args': (output_paths[args.dataset]['statement']['dev'],
            #           output_paths[args.dataset]['tokenized']['dev'])},
            # {'func': tokenize_statement_file,
            #  'args': (output_paths[args.dataset]['statement']['test'],
            #           output_paths[args.dataset]['tokenized']['test'])},

            {'func': make_word_vocab,
             'args': ((output_paths['dataset']['statement']['train'].format(dataset=args.dataset),),
                      output_paths['dataset']['statement']['vocab'].format(dataset=args.dataset))},

            # Співставляються слова які є ентіті ConceptNet графу зі словами які знаходяться в реченні.
            # Виділяється окремо ентіті з запитання та ентіті з відповіді, куди потрапляють всі слова які присутні в
            # графі.
            {'func': ground,
             'args': (output_paths['dataset']['statement']['train'].format(dataset=args.dataset),
                      output_paths['graph']['vocab'].format(graph=args.graph),
                      output_paths['graph']['patterns'].format(graph=args.graph),
                      output_paths['dataset']['grounded']['train'].format(dataset=args.dataset),
                      args.nprocs)},
            {'func': ground,
             'args': (output_paths['dataset']['statement']['dev'].format(dataset=args.dataset),
                      output_paths['graph']['vocab'].format(graph=args.graph),
                      output_paths['graph']['patterns'].format(graph=args.graph),
                      output_paths['dataset']['grounded']['dev'].format(dataset=args.dataset),
                      args.nprocs)},
            {'func': ground,
             'args': (output_paths['dataset']['statement']['test'].format(dataset=args.dataset),
                      output_paths['graph']['vocab'].format(graph=args.graph),
                      output_paths['graph']['patterns'].format(graph=args.graph),
                      output_paths['dataset']['grounded']['test'].format(dataset=args.dataset),
                      args.nprocs)},

            # Знаходяться найближчі 100 шляхів між ентіті з питання та ентіті з відповіді. Для кожного шляху
            # визначається перелік рілейшенів.
            # {'func': find_paths,
            #  'args': (output_paths['dataset']['grounded']['train'],
            #           output_paths['graph']['vocab'],
            #           output_paths['graph']['pruned-graph'],
            #           output_paths['dataset']['paths']['raw-train'],
            #           args.nprocs,
            #           args.seed)},
            # {'func': find_paths,
            #  'args': (output_paths['dataset']['grounded']['dev'],
            #           output_paths['graph']['vocab'],
            #           output_paths['graph']['pruned-graph'],
            #           output_paths['dataset']['paths']['raw-dev'],
            #           args.nprocs,
            #           args.seed)},
            # {'func': find_paths,
            #  'args': (output_paths['dataset']['grounded']['test'],
            #           output_paths['graph']['vocab'],
            #           output_paths['graph']['pruned-graph'],
            #           output_paths['dataset']['paths']['raw-test'],
            #           args.nprocs,
            #           args.seed)},

            # Побудова скорінгу шляхів базуючись на ембедінгах transe. Ті шляхи кращі, які мають меншу косинусну
            # відстань emb_rel від emb_head - emb_tail
            # {'func': score_paths,
            #  'args': (output_paths['dataset']['paths']['raw-train'],
            #           input_paths['dataset']['transe']['ent'],
            #           input_paths['dataset']['transe']['rel'],
            #           output_paths['graph']['vocab'],
            #           output_paths['dataset']['paths']['scores-train'], args.nprocs)},
            # {'func': score_paths,
            #  'args': (output_paths['dataset']['paths']['raw-dev'],
            #           input_paths['dataset']['transe']['ent'],
            #           input_paths['dataset']['transe']['rel'],
            #           output_paths['graph']['vocab'],
            #           output_paths['dataset']['paths']['scores-dev'],
            #           args.nprocs)},
            # {'func': score_paths,
            #  'args': (output_paths['dataset']['paths']['raw-test'],
            #           input_paths['dataset']['transe']['ent'],
            #           input_paths['dataset']['transe']['rel'],
            #           output_paths['graph']['vocab'],
            #           output_paths['dataset']['paths']['scores-test'],
            #           args.nprocs)},

            # Фільтрація шляхів згідно отриманих score. Якщо score більше ніж args.path_prune_threshold, то даний шлях
            # лишається.
            # {'func': prune_paths,
            #  'args': (output_paths['dataset']['paths']['raw-train'],
            #           output_paths['dataset']['paths']['scores-train'],
            #           output_paths['dataset']['paths']['pruned-train'],
            #           args.path_prune_threshold)},
            # {'func': prune_paths,
            #  'args': (output_paths['dataset']['paths']['raw-dev'],
            #           output_paths['dataset']['paths']['scores-dev'],
            #           output_paths['dataset']['paths']['pruned-dev'],
            #           args.path_prune_threshold)},
            # {'func': prune_paths,
            #  'args': (output_paths['dataset']['paths']['raw-test'],
            #           output_paths['dataset']['paths']['scores-test'],
            #           output_paths['dataset']['paths']['pruned-test'],
            #           args.path_prune_threshold)},

            # Compose graph from entities presented in paths.
            # - Окрім шляхів між question концепцій (head) та answer концепцій (tail), знаходяться шляхи між концепцій
            #   всередині question / answer, якщо такі безпосередні зв'язки присутні (зв'язки без проміжних сутностей).
            # - В отриманому графі ресетяться індекси, але старі індекси зберігаються в діктах під ключем cid.
            # {'func': generate_graph,
            #  'args': (output_paths['dataset']['grounded']['train'],
            #           output_paths['dataset']['paths']['pruned-train'],
            #           output_paths['graph']['vocab'],
            #           output_paths['graph']['pruned-graph'],
            #           output_paths['dataset']['graph']['train'])},
            # {'func': generate_graph,
            #  'args': (output_paths['dataset']['grounded']['dev'],
            #           output_paths['dataset']['paths']['pruned-dev'],
            #           output_paths['graph']['vocab'],
            #           output_paths['graph']['pruned-graph'],
            #           output_paths['dataset']['graph']['dev'])},
            # {'func': generate_graph,
            #  'args': (output_paths['dataset']['grounded']['test'],
            #           output_paths['dataset']['paths']['pruned-test'],
            #           output_paths['graph']['vocab'],
            #           output_paths['graph']['pruned-graph'],
            #           output_paths['dataset']['graph']['test'])},

            # Побудова матриці суміжності між ентіті з питань, відповідей та попарно спільно сусідніх до них нод.
            # Матриця суміжності має шейп (n_rels, n_nodes, n_nodes). Окрім неї зберігається масив з переліком концепцій
            # та масив масок для позиціонування концептів з питання та концептів з відповіді.
            {'func': generate_adj_data_from_grounded_concepts,
             'args': (output_paths['dataset']['grounded']['train'].format(dataset=args.dataset),
                      output_paths['graph']['pruned-graph'].format(graph=args.graph),
                      output_paths['graph']['vocab'].format(graph=args.graph),
                      # для кожної пари питання-відповідь, надається матриця суміжності, масив концептів, маска
                      # концептів з питання та маска концептів з відповіді
                      output_paths['dataset']['graph']['adj-train'].format(dataset=args.dataset),
                      args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts,
             'args': (output_paths['dataset']['grounded']['dev'].format(dataset=args.dataset),
                      output_paths['graph']['pruned-graph'].format(graph=args.graph),
                      output_paths['graph']['vocab'].format(graph=args.graph),
                      output_paths['dataset']['graph']['adj-dev'].format(dataset=args.dataset),
                      args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts,
             'args': (output_paths['dataset']['grounded']['test'].format(dataset=args.dataset),
                      output_paths['graph']['pruned-graph'].format(graph=args.graph),
                      output_paths['graph']['vocab'].format(graph=args.graph),
                      output_paths['dataset']['graph']['adj-test'].format(dataset=args.dataset),
                      args.nprocs)},

            # Конвертація матриці суміжності що була обчислена кроком вище в перелік триплетів.
            # Тріплети відсортовані таким чином, що спочатку йдуть ті в яких обидва ентіті знаходяться в
            # mentioned_concepts (були зазначені в питанні чи відповіді). Далі де зазначені концепти є або в хеді, або
            # в тейлі. Далі йдуть ті триплети в яких немає ентіті ні в хеді ні в тейлі. Значення що зіпається з даним
            # масивом триплетів визначає кількість триплетів з перших двох категорій.
            # До третьої категорії відходять тріплети у яких і хеда і теіл - це ентіті з сусідніх до mentioned concepts.
            {'func': generate_triples_from_adj,
             'args': (output_paths['dataset']['graph']['adj-train'].format(dataset=args.dataset),
                      output_paths['dataset']['grounded']['train'].format(dataset=args.dataset),
                      output_paths['graph']['vocab'].format(graph=args.graph),
                      output_paths['dataset']['triple']['train'].format(dataset=args.dataset))},
            {'func': generate_triples_from_adj,
             'args': (output_paths['dataset']['graph']['adj-dev'].format(dataset=args.dataset),
                      output_paths['dataset']['grounded']['dev'].format(dataset=args.dataset),
                      output_paths['graph']['vocab'].format(graph=args.graph),
                      output_paths['dataset']['triple']['dev'].format(dataset=args.dataset))},
            {'func': generate_triples_from_adj,
             'args': (output_paths['dataset']['graph']['adj-test'].format(dataset=args.dataset),
                      output_paths['dataset']['grounded']['test'].format(dataset=args.dataset),
                      output_paths['graph']['vocab'].format(graph=args.graph),
                      output_paths['dataset']['triple']['test'].format(dataset=args.dataset))},

            {'func': generate_path_and_graph_from_adj,
             'args': (output_paths['dataset']['graph']['adj-train'].format(dataset=args.dataset),
                      output_paths['graph']['pruned-graph'].format(graph=args.graph),
                      output_paths['dataset']['paths']['adj-train'].format(dataset=args.dataset),
                      output_paths['dataset']['graph']['nxg-from-adj-train'].format(dataset=args.dataset),
                      args.nprocs)},
            {'func': generate_path_and_graph_from_adj,
             'args': (output_paths['dataset']['graph']['adj-dev'].format(dataset=args.dataset),
                      output_paths['graph']['pruned-graph'].format(graph=args.graph),
                      output_paths['dataset']['paths']['adj-dev'].format(dataset=args.dataset),
                      output_paths['dataset']['graph']['nxg-from-adj-dev'].format(dataset=args.dataset),
                      args.nprocs)},
            {'func': generate_path_and_graph_from_adj,
             'args': (output_paths['dataset']['graph']['adj-test'].format(dataset=args.dataset),
                      output_paths['graph']['pruned-graph'].format(graph=args.graph),
                      output_paths['dataset']['paths']['adj-test'].format(dataset=args.dataset),
                      output_paths['dataset']['graph']['nxg-from-adj-test'].format(dataset=args.dataset),
                      args.nprocs)},
        ]
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
