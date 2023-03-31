import pickle
import time
import numpy as np
import os
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn import metrics
from new_maaf_common import read_datasets
from sota_util import Dictionary
from sota_model import FSNet


def ArgumentParser():

    parser = argparse.ArgumentParser()
    # general parameters
    # activation selection: "relu", "selu"
    parser.add_argument("--learning_rate", default=0.0005,type=float,
                        help="learning rate")
    parser.add_argument("--epochs", default=30,type=int,
                        help="training epochs")
    parser.add_argument("--batch_size", default=128,type=int,
                        help="training and eval batch size")
    parser.add_argument("--max_length", default=32,type=int,
                        help="packle length")
    parser.add_argument("--seed", default=1,type=int,
                        help="torch seed")
    parser.add_argument("--cuda", default=True, type=bool,
                        help="weather to use cuda")
    parser.add_argument("--gpu_id", default=1, type=int,
                        help="gpu choose")
    parser.add_argument("--log_interval", default=50, type=int,
                        help="log_interval")

    # fs_net model parameters
    parser.add_argument("--embedding_size", default=128,type=int,
                        help="fs net embedding size")
    parser.add_argument("--hidden_unit", default=128,type=int,
                        help="fs net hidden unit")
    parser.add_argument("--alpha", default=1.0,type=float,
                        help="fs net alpha")
    parser.add_argument("--layers_num", default=2,type=int,
                        help="fs net layers num")
    parser.add_argument("--dropout", default=0.3,type=float,
                        help="fs net dropout")
    return parser.parse_args()


def get_params(train_ratio=0.6, validate_ratio=0.1, test_ratio=0.3, train_data='data-02', test_data='data-02', packet_limit=0):
    params = {'train_ratio': train_ratio, 'validate_ratio': validate_ratio, 'test_ratio': test_ratio,
              'train_data': train_data, 'test_data': test_data, 'packet_limit': packet_limit}
    return params


def check_result_file_existance(log_dir, random_time):

    log_filename = 'fsnet_log_' + str(random_time) + '.txt'
    log_path = os.path.join(log_dir, log_filename)

    confuse_matrix_filename = 'fsnet_confuse_matrix_' + str(random_time) + '.npy'
    confuse_matrix_path = os.path.join(log_dir, confuse_matrix_filename)

    log_exist = os.path.exists(log_path)
    confuse_matrix_exist = os.path.exists(confuse_matrix_path)

    if log_exist and confuse_matrix_exist:
        return True
    else:
        return False


def get_param_str(params):
    params_dir_name = \
        'train_rat-' + str(params['train_ratio']) + '-' + \
        'valid_rat-' + str(params['validate_ratio']) + '-' + \
        'test_rat-' + str(params['test_ratio']) + '-' + \
        'train-' + params['train_data'] + '-' + \
        'test-' + params['test_data']
    if params['packet_limit'] > 0:
        params_dir_name += '-packet_limit-' + str(params['packet_limit'])
    return params_dir_name


def get_result_dir(result_dir, params):
    params_dir_name = get_param_str(params)
    ml_result_dir = os.path.join(result_dir, params_dir_name)
    os.makedirs(ml_result_dir, exist_ok=True)

    return ml_result_dir


def data_construct(params, random_time):
    train_ratio = params['train_ratio']
    validate_ratio = params['validate_ratio']
    test_ratio = params['test_ratio']

    # shuffle and divide dataset
    if params['train_data'] == params['test_data'] == 'data-02':
        dataset = datasets['data-02']
        dataset_keys = list(dataset.keys())
        rand = random.Random(random_time)
        rand.shuffle(dataset_keys)

        train_data_keys = dataset_keys[:int(len(dataset_keys)*train_ratio)]
        validate_data_keys = dataset_keys[int(len(dataset_keys)*train_ratio):int(len(dataset_keys)*(train_ratio+validate_ratio))]
        test_data_keys = dataset_keys[int(len(dataset_keys)*(train_ratio+validate_ratio)):]

        train_data = [dataset[key] for key in train_data_keys]
        validate_data = [dataset[key] for key in validate_data_keys]
        test_data = [dataset[key] for key in test_data_keys]

    elif params['train_data'] == 'data-02' and params['test_data'] == 'monkeydata-02':
        data02_dataset = datasets['data-02']
        monkeydata02_dataset = datasets['monkeydata-02']
        data02_dataset_keys = list(data02_dataset.keys())
        monkeydata02_dataset_keys = list(monkeydata02_dataset.keys())

        train_ratio_new = train_ratio / (train_ratio + validate_ratio)
        train_data_keys = data02_dataset_keys[:int(len(data02_dataset_keys)*train_ratio_new)]
        validate_data_keys = data02_dataset_keys[int(len(data02_dataset_keys)*train_ratio_new):]
        test_data_keys = monkeydata02_dataset_keys

        train_data = [data02_dataset[key] for key in train_data_keys]
        validate_data = [data02_dataset[key] for key in validate_data_keys]
        test_data = [monkeydata02_dataset[key] for key in test_data_keys]

    elif params['train_data'] == 'monkeydata-02' and params['test_data'] == 'data-02':
        data02_dataset = datasets['data-02']
        monkeydata02_dataset = datasets['monkeydata-02']
        data02_dataset_keys = list(data02_dataset.keys())
        monkeydata02_dataset_keys = list(monkeydata02_dataset.keys())

        train_ratio_new = train_ratio / (train_ratio + validate_ratio)
        train_data_keys = monkeydata02_dataset_keys[:int(len(monkeydata02_dataset_keys)*train_ratio_new)]
        validate_data_keys = monkeydata02_dataset_keys[int(len(monkeydata02_dataset_keys)*train_ratio_new):]
        test_data_keys = data02_dataset_keys

        train_data = [monkeydata02_dataset[key] for key in train_data_keys]
        validate_data = [monkeydata02_dataset[key] for key in validate_data_keys]
        test_data = [data02_dataset[key] for key in test_data_keys]

    else:
        print('dataset error')
        raise KeyError

    return train_data, validate_data, test_data


def fsnet_vector(dataset, app_list, args, params):
    packet_limit = params['packet_limit']
    data_vector = []
    for stream in dataset:
        app_index = app_list.index(stream['app'])
        # message_length_seq = [message[1] for message in stream['message_sequence']]
        # message_length_seq = message_length_seq[:args.max_length]
        # message_length_seq = message_length_seq[:args.max_length] + [0] * (args.max_length - len(message_length_seq))
        # data_vector.append([app_index] + message_length_seq)
        packet_length_seq = [packet[2] for packet in stream['raw_packet_sequence']]
        if packet_limit > 0:
            packet_length_seq = packet_length_seq[:packet_limit]
        packet_length_seq = packet_length_seq[:args.max_length]
        data_vector.append([app_index] + packet_length_seq)
    return data_vector


def data_package(data, dictionary, max_len):
    """Package data for training / evaluation."""
    dat = [[dictionary.word2idx[packet_len] for packet_len in len_seq[1:]] for len_seq in data]
    targets = [len_seq[0] for len_seq in data]

    for i in range(len(dat)):
        if max_len < len(dat[i]):
            dat[i] = dat[i][:max_len]
        else:
            dat[i].extend((max_len - len(dat[i])) * [dictionary.word2idx[0]])

    dat = torch.LongTensor(dat)
    targets = torch.LongTensor(targets)
    # return dat.t(), targets
    # return dat.transpose(0, 1), targets
    return dat, targets


def evaluate(model, test_vector, dictionary, args):
    """evaluate the model while training"""
    model.eval()  # turn on the eval() switch to disable dropout
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_correct = 0

    prediction_all = torch.zeros(size=(0,), dtype=torch.int)
    target_all = torch.zeros(size=(0,), dtype=torch.int)
    if args.cuda:
        prediction_all= prediction_all.cuda()
        target_all = target_all.cuda()

    for batch, i in enumerate(range(0, len(test_vector), args.batch_size)):
        batch_data, targets = data_package(test_vector[i:i+args.batch_size], dictionary, args.max_length)
        if args.cuda:
            batch_data = batch_data.cuda()
            targets = targets.cuda()
        _, output = model.forward(batch_data)

        total_loss += criterion(output, targets).data
        prediction = torch.max(output, 1)[1]
        total_correct += torch.sum((prediction == targets).float())

        prediction_all = torch.cat([prediction_all, prediction], dim=0)
        target_all = torch.cat([target_all, targets], dim=0)
    return total_loss / (len(test_vector) // args.batch_size), total_correct / len(test_vector), prediction_all, target_all


def train(model, epoch_number, train_vector, test_vector, dictionary, args):
    # init training
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    total_loss = 0

    # batch loop
    start_time = time.time()
    for batch, i in enumerate(range(0, len(train_vector), args.batch_size)):
        # forward
        batch_data, targets = data_package(train_vector[i:i+args.batch_size], dictionary, args.max_length)
        if args.cuda:
            batch_data = batch_data.cuda()
            targets = targets.cuda()
        recon, output = model.forward(batch_data)

        # loss calculation
        recon_loss = criterion(recon, targets)
        output_loss = criterion(output, targets)
        loss = recon_loss + args.alpha * output_loss
        total_loss += loss.data

        # back forward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % args.log_interval == 0 and batch > 0:
            elapsed = time.time() - start_time
            fmt = '| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.4f}'
            print(fmt.format(epoch_number, batch, len(train_vector) // args.batch_size,
                  elapsed * 1000 / args.log_interval, total_loss / args.log_interval))
            total_loss = 0
            start_time = time.time()

    # evaluate_start_time = time.time()
    # evaulate_loss, acc = evaluate(model, train_vector, dictionary, args)
    # print('-' * 89)
    # fmt = '| evaluation data_train | time: {:5.2f}s | valid loss (pure) {:5.4f} | Acc {:8.4f}'
    # print(fmt.format((time.time() - evaluate_start_time), evaulate_loss, acc))

    evaluate_start_time = time.time()
    evaulate_loss, acc, prediction, target = evaluate(model, test_vector, dictionary, args)
    print('-' * 89)
    fmt = '| evaluation data_val   | time: {:5.2f}s | valid loss (pure) {:5.4f} | Acc {:8.4f}'
    print(fmt.format((time.time() - evaluate_start_time), evaulate_loss, acc))
    print('-' * 89)

    return prediction, target


def fsnet_dl(model, app_list, random_time, train_vector, test_vector, ml_result_dir, args):

    # start model training
    for epoch in range(args.epochs):
        prediction, target = train(model, epoch, train_vector, test_vector, dictionary, args)

    prediction = prediction.cpu().numpy()
    target = target.cpu().numpy()

    print(metrics.classification_report(y_pred=prediction, y_true=target, target_names=app_list, digits=4))
    report = metrics.classification_report(y_pred=prediction, y_true=target, target_names=app_list, output_dict=True)
    confuse_matrix = metrics.confusion_matrix(y_pred=prediction, y_true=target)

    log_dir = ml_result_dir
    os.makedirs(log_dir, exist_ok=True)

    log_filename = 'fsnet_log_' + str(random_time) + '.txt'
    log_path = os.path.join(log_dir, log_filename)
    with open(log_path, 'w') as log_fp:
        log_fp.write(json.dumps(report))

    confuse_matrix_filename = 'fsnet_confuse_matrix_' + str(random_time) + '.npy'
    confuse_matrix_path = os.path.join(log_dir, confuse_matrix_filename)
    np.save(confuse_matrix_path, confuse_matrix)


if __name__ == '__main__':
    # read datasets
    dataset_dir = 'data-pk'
    dataset_names = ['data-02', 'monkeydata-02']
    app_list = ['alipay', 'taobao', 'amap', 'baidusearchbox', 'baidumap', 'facebook', 'instagram', 'twitter', 'weibo',
                'airbnb', 'linkedin', 'evernote', 'blued', 'ele', 'github', 'yirendai']
    dataset_cache_dir = 'data-cache'
    datasets = read_datasets(app_list, dataset_dir, dataset_names, dataset_cache_dir)
    print('Load Dataset Finished')

    # configure
    args = ArgumentParser()
    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)
            torch.cuda.set_device(args.gpu_id)
    random.seed(args.seed)

    fsnet_result_dir = os.path.join('result', 'fsnet')
    random_times = 10

    # set parameters
    params_list = []

    params = get_params(train_data='data-02', test_data='data-02')
    if params not in params_list:
        params_list.append(params)

    # study cross dataset
    params = get_params(train_data='data-02', test_data='monkeydata-02')
    if params not in params_list:
        params_list.append(params)
    params = get_params(train_data='monkeydata-02', test_data='data-02')
    if params not in params_list:
        params_list.append(params)

    # dataset partition
    dataset_patitions = [(0.1, 0.1, 0.8), (0.2, 0.1, 0.7), (0.3, 0.1, 0.6), (0.4, 0.1, 0.5), (0.5, 0.1, 0.4), (0.6, 0.1, 0.3), (0.7, 0.1, 0.2), (0.8, 0.1, 0.1)]
    for train_ratio, validate_ratio, test_ratio in dataset_patitions:
        params = get_params(train_ratio=train_ratio, validate_ratio=validate_ratio, test_ratio=test_ratio, train_data='data-02', test_data='data-02')
        if params not in params_list:
            params_list.append(params)

    # packet limit maybe start from 1
    for packet_limit in range(1, 51):
        params = get_params(train_data='data-02', test_data='data-02', packet_limit=packet_limit)
        if params not in params_list:
            params_list.append(params)

    # start running
    for params in params_list:
        print('Start Task with Params:', params)
        result_dir = get_result_dir(fsnet_result_dir, params)

        for random_time in range(random_times):
            # check result existence
            if check_result_file_existance(result_dir, random_time):
                continue
                # pass

            print('Random Time', random_time, 'Started')

            # data construction
            train_data, validate_data, test_data = data_construct(params, random_time)
            print('Random Time', random_time, 'Data Construction Finished')

            # data preprocessing
            train_vector = fsnet_vector(train_data + validate_data, app_list, args, params)
            test_vector = fsnet_vector(test_data, app_list, args, params)
            dictionary = Dictionary(train_data + validate_data + test_data)
            n_token = len(dictionary)
            print('Random Time', random_time, 'FSNet Data Preprocessing Finished')

            # model init
            model = FSNet(args, n_token, dictionary, app_list)
            if args.cuda:
                model = model.cuda()

            # model training and evaluating
            fsnet_dl(model, app_list, random_time, train_vector, test_vector, result_dir, args)
            print('Random Time', random_time, 'FSNet Deep Learning Finished')