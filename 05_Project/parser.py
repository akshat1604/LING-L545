'''
Author: Geeticka Chauhan
Parser for hyperparameters with the default ones used in this work. 
'''


import argparse
import copy
import time

#Arguments that should be provided: dataset: when extending to other datasets this will be useful
#                                   use_elmo: increases runtime so just for testing don't use
# Arguments that are commonly provided: cross validate, use_test, use_piecewise_pool
# boolean arguments should be provided like --cross_validate without any other value
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='semeval2010',
        help='the dataset for which the task is being applied. Options: semeval2010, ddi')
parser.add_argument('--id', default='baseline', # this will get overwritten in the code by uuid 
                                        help="a name for identifying the model")
parser.add_argument('--pos_embed_size', default=25, type=int,
                                        help="position feature embedding size")
# parser.add_argument('--pos_embed_size_dep', default=42, type=int,
#                                         help="position feature embedding size for dependency path")
# parser.add_argument('--pos_tag_embed_size', default=5, type=int,
#                                         help="pos tag embedding size")
parser.add_argument('--pos_embed_num', default=123, type=int,
                                        help="number of positions")
# parser.add_argument('--pos_embed_num_dep', default=123, type=int,
#                                         help='number of positions for dep positions')
parser.add_argument('--classnum', default=18, type=int,
                                        help="number of output class")
parser.add_argument('--num_filters', default=100, type=int,
                                        help="number of convolution filters")
# parser.add_argument('--num_filters_dep', default=300, type=int,
#                                         help="number of convolution filters for dependency")
parser.add_argument('--filter_sizes', default='2,3,4,5', type=str,
                                        help='convolution window size')
# parser.add_argument('--filter_sizes_dep', default='2,3,4', type=str,
#                                         help='convolution window size for dependency')
parser.add_argument('--batch_size', default=50, type=int,
                                        help='input batch size')
parser.add_argument('--keep_prob', default=0.5, type=float,
                                        help='dropout probability')
parser.add_argument('--l2_reg_lambda', default=0.005, type=float,
                                        help='l2 regularizaition ratio')
parser.add_argument('--m_plus', default=2.5, type=float,
                                        help='positive margin needed by loss function')
parser.add_argument('--m_minus', default=0.5, type=float,
                                        help='negative margin needed by loss function')
parser.add_argument('--gamma', default=2, type=float,
                                        help='scaling factor for the loss')
parser.add_argument('--border_size', default=50, type=int,
                                        help='how many tokens beyond the two entities included')
parser.add_argument('--pickle_seed', default=5, type=int,
                                        help='the seed to use for the input pickle file')
parser.add_argument('--seed', default=1, type=int,
                                        help='the seed to set for the tensorflow graph for reproducibility')

# parser.add_argument('--grad_clipping', default=0.01, type=float, \
#                                                 help='parameter used by gradient clipping')
parser.add_argument('--early_stop', default=False, action='store_true',
                                                help='whether to do early stop')
parser.add_argument('--low_freq_thresh', default=0, type=int,
                                                 help='what frequency of word to send to 0 index')

# Optimization related arguments
parser.add_argument('--num_epoches', default=250, type=int,
                                        help='number of epochs')
parser.add_argument('--patience', default=100, type=int,
                                        help='number of epochs to wait before early stop if no progress')
parser.add_argument('--lr_values', default=[0.001, 0.001, 0.001], nargs="*",type=float,
                                        help='learning rate values for the piecewise constant')
parser.add_argument('--lr_boundaries', default=[60,80], nargs="*",type=int,
                                        help='learning rate boundaries for the piecewise constant')
parser.add_argument('--momentum', default=0.9, type=float, \
                                                help='momentum for SGD with momentum; only valid if' +\
                                                'sgd_momentum is True')
parser.add_argument('--sgd_momentum', default=False, action='store_true', \
                                                help='whether to use SGD with momentum')
# Misc arguments
parser.add_argument('--save_graph_dir', default=None,
                                                help='save graphs here')
parser.add_argument('--embedding_file', default='/data/medg/misc/geeticka/relation_extraction/senna/embeddings.txt',
                                                help='embedding file')
parser.add_argument('--embedding_vocab', default='/data/medg/misc/geeticka/relation_extraction/senna/words.lst',
                                                help='embedding vocab file')
parser.add_argument('--data_root',
        default="sample.csv",
                                                help= "Data root directory")
parser.add_argument('--train_text_dataset_path', default='train_original.txt',
                                                help='test file')
parser.add_argument('--test_text_dataset_path', default='test_original.txt',
                                                help='test file')
parser.add_argument('--dev_answer_file', default='answers_for_dev.txt',
                                                help='dev answer file')
parser.add_argument('--test_answer_file', default='answers_for_test.txt',
                                                help='test answer file')
parser.add_argument('--result_file', default='result.txt',
                                                help='result file of macro-f1 checking')
parser.add_argument('--output_dir', default='/data/scratch-oc40/geeticka/relation_extraction/output/',
                                                help='output directory')
parser.add_argument('--tensorboard_folder', default='output/Tensorboard', \
                                                help='tensorboard output directory')
parser.add_argument('--output_folder', default=None, help='model specific output directory')
parser.add_argument('--result_folder', default='output/Result', help='Specific'\
                                                + 'result folder for all folds with avg and std for a hyperparam setting')
parser.add_argument('--final_result_folder', default=None, help="Final result CSV file for all runs")
parser.add_argument('--log_file', default=None,
                                                help='log file')

parser.add_argument('--early_stop_size', default=0.5, type=float,
                                                help='early stop size as a percentage of train set')
parser.add_argument('--cross_validate', default=False, action='store_true',
                                                help='whether to implement cross validation (use for hyperparam tuning')
parser.add_argument('--cross_validate_report', default=False, action='store_true',
                                                help='whether to implement cross validation for reporting purposes')
parser.add_argument('--use_test', default=False, action='store_true',
                                                help='use the full train and test data split')
parser.add_argument('--use_elmo', default=False, action='store_true',
                                                help='whether to use the elmo embeddings')
parser.add_argument('--use_bert_CLS', default=False, action='store_true',
                                                help='whether to use the bert embeddings CLS tokens')
parser.add_argument('--use_bert_tokens', default=False, action='store_true',
                                                help='whether to use the bert embeddings individual tokens')
parser.add_argument('--use_piecewise_pool', default=False, action='store_true',
                                                help='whether to do piecewise max pooling')
parser.add_argument('--hyperparam_tuning_mode', default=False, action='store_true',
                                                help='whether hyperparameter tuning mode was on')
parser.add_argument('--random_search', default=False, action='store_true',
                                                help='whether random search should be implemented')
parser.add_argument('--preprocessing_type', default='original', 
help= 'specify the preprocessing type from original, entity_blinding, punct_digit, punct_stop_digit')

#TODO (geeticka) : below is not needed to include in parser
parser.add_argument('--fold', default=None, type=int,\
                                                help='the current fold')
# TODO(geeticka): Don't add things to argparse unless you want to set them on the command line.
parser.add_argument('--eval_metric_folds', default=[], type=list,\
                                                help='the list of macro f1s for all folds if folds exist')
parser.add_argument('--add_hyperparam_details', default=None, type=str, \
                                                help='a string that contains information about miscellaneous'\
                                                + 'hyperparameter details')
parser.add_argument('--train_start_folds', default=[], type=list, \
                                                help='start times of each of the models by fold')
parser.add_argument('--execution_time_folds', default=[], type=list, \
                                                help='execution times of each of the models by fold')

#config = parser.parse_args()
#TODO (geeticka): will need to have another parameter for the hyperparameter tuning case so that you can dump the csv for hyperparam tuning as a separate csv

def get_config(): return parser.parse_args()
def get_results_dict(config, train_start_time_in_miliseconds):
    results = {} # this dictionary will contain all the result of the experiment    results['model_options'] = copy.copy(model_options)
    parameters = {}
    parameters['dataset'] = config.dataset
    parameters['preprocessing_type'] = config.preprocessing_type
    parameters['id'] = config.id
    parameters['pos_embed_size'] = config.pos_embed_size
    parameters['pos_embed_num'] = config.pos_embed_num
    parameters['num_filters'] = config.num_filters
    parameters['filter_sizes'] = config.filter_sizes
    parameters['batch_size'] = config.batch_size
    parameters['keep_prob'] = config.keep_prob
    parameters['l2_reg_lambda'] = config.l2_reg_lambda
    parameters['m_plus'] = config.m_plus
    parameters['m_minus'] = config.m_minus
    parameters['gamma'] = config.gamma
    parameters['border_size'] = config.border_size
    parameters['pickle_seed'] = config.pickle_seed
    parameters['seed'] = config.seed
    parameters['early_stop'] = config.early_stop
    #optimization related arguments
    parameters['num_epoches'] = config.num_epoches
    parameters['patience'] = config.patience # Number of epoch to wait before early stop if no progress
    parameters['lr_values'] = config.lr_values
    parameters['lr_boundaries'] = config.lr_boundaries
    parameters['momentum'] = config.momentum
    parameters['sgd_momentum'] = config.sgd_momentum

    parameters['embedding_file'] = config.embedding_file
    parameters['embedding_vocab'] = config.embedding_vocab

    parameters['early_stop_size'] = config.early_stop_size
    parameters['cross_validate'] = config.cross_validate
    parameters['cross_validate_report'] = config.cross_validate_report
    parameters['use_test'] = config.use_test
    parameters['use_elmo'] = config.use_elmo
    parameters['use_bert_CLS'] = config.use_bert_CLS
    parameters['use_bert_tokens'] = config.use_bert_tokens
    parameters['use_piecewise_pool'] = config.use_piecewise_pool
    #parameters['dev_size'] = config.dev_size
    #parameters['use_lemmas'] = config.use_lemmas
    # parameters['fold'] = config.fold
    results['model_options'] = copy.copy(parameters)
    results['execution_details'] = {}
    results['execution_details']['train_start'] = time.time()
    results['execution_details']['time_stamp'] = train_start_time_in_miliseconds
    results['execution_details']['early_stop'] = False
    results['execution_details']['keyboard_interrupt'] = False
    results['execution_details']['num_epochs'] = 0 # TODO (geeticka): legacy parameter - delete this 

    return results, parameters
