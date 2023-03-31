import os
import gzip
import pickle


def get_params(train_ratio=0.6, validate_ratio=0.1, test_ratio=0.3,
               dns=False, common_name=False, org_name=False, sni=False, cipher_suite=False, ref_weight='stream',
               length_sequence='format_message', length_count=0, window_size=False, ml_method='xgboost',
               train_data='data-02', test_data='data-02', unknown_test=False, packet_limit=0):
    params = {'train_ratio': train_ratio, 'validate_ratio': validate_ratio, 'test_ratio': test_ratio,
              'dns': dns, 'common_name': common_name, 'org_name': org_name, 'sni': sni,
              'cipher_suite': cipher_suite, 'ref_weight': ref_weight, 'length_sequence': length_sequence,
              'length_count': length_count, 'window_size': window_size, 'ml_method': ml_method,
              'train_data': train_data, 'test_data': test_data, 'unknown_test': unknown_test, 'packet_limit': packet_limit}
    return params

def get_best_params(train_ratio=0.6, validate_ratio=0.1, test_ratio=0.3,
               dns=True, common_name=True, org_name=True, sni=True, cipher_suite=True, ref_weight='stream',
               length_sequence='format_message', length_count=30, window_size=False, ml_method='xgboost',
               train_data='data-02', test_data='data-02', unknown_test=False, packet_limit=0):
    params = {'train_ratio': train_ratio, 'validate_ratio': validate_ratio, 'test_ratio': test_ratio,
              'dns': dns, 'common_name': common_name, 'org_name': org_name, 'sni': sni,
              'cipher_suite': cipher_suite, 'ref_weight': ref_weight, 'length_sequence': length_sequence,
              'length_count': length_count, 'window_size': window_size, 'ml_method': ml_method,
              'train_data': train_data, 'test_data': test_data, 'unknown_test': unknown_test, 'packet_limit': packet_limit}
    return params


def check_result_file_existance(ml_result_dir, ml_method, random_time):
    ml_method = ml_method.replace('_', '')

    log_dir = os.path.join(ml_result_dir)
    log_filename = ml_method + '_log_' + str(random_time) + '.txt'
    log_path = os.path.join(log_dir, log_filename)

    confuse_matrix_filename = ml_method + '_confuse_matrix_' + str(random_time) + '.npy'
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
        'dns-' + str(params['dns']) + '-' + \
        'common-' + str(params['common_name']) + '-' + \
        'org-' + str(params['org_name']) + '-' + \
        'sni-' + str(params['sni']) + '-' + \
        'cipher-' + str(params['cipher_suite']) + '-' + \
        'ref-' + params['ref_weight'] + '-' + \
        'length-' + params['length_sequence'] + '-' + \
        'count-' + str(params['length_count']) + '-' + \
        'window-' + str(params['window_size']) + '-' + \
        'method-' + params['ml_method'] + '-' + \
        'train-' + params['train_data'] + '-' + \
        'test-' + params['test_data'] + '-' + \
        'unknown-' + str(params['unknown_test'])
    if params['packet_limit'] > 0:
        params_dir_name += '-packet_limit-' + str(params['packet_limit'])
    return params_dir_name


def get_result_dir(new_maaf_result_dir, params):

    params_dir_name = get_param_str(params)
    ml_result_dir = os.path.join(new_maaf_result_dir, params_dir_name)
    os.makedirs(ml_result_dir, exist_ok=True)

    return ml_result_dir


def read_datasets(app_list, data_dir, dataset_names, dataset_cache_dir):
    dataset_cache_path = os.path.join(dataset_cache_dir, 'data-cache.pk')
    if os.path.exists(dataset_cache_path):
        with open(dataset_cache_path, 'rb') as fp:
            datasets = pickle.load(fp)

    else:
        datasets = dict()
        # read datasets
        for dataset_name in dataset_names:
            datasets[dataset_name] = dict()
            dataset_dir = os.path.join(data_dir, dataset_name)
            for app in app_list:
                app_dir = os.path.join(dataset_dir, app)
                pk_file_names = os.listdir(app_dir)
                for pk_file_name in pk_file_names:
                    if pk_file_name.endswith('.pk.gz'):
                        pk_file_path = os.path.join(app_dir, pk_file_name)
                        print('Reading', pk_file_path)
                        with gzip.open(pk_file_path, 'rb') as fp:
                            streams = pickle.load(fp)
                            for id, stream in streams.items():
                                if isinstance(stream['dns'], dict):
                                    dns = stream['dns']['domain_name']
                                else:
                                    dns = None

                                # remove google related streams
                                if dns:
                                    if 'google' in dns:
                                        continue
                                if stream['sni']:
                                    if 'google' in stream['sni']:
                                        continue
                                if stream['common_name']:
                                    if 'google' in stream['common_name']:
                                        continue

                                datasets[dataset_name][id] = {
                                    'app': stream['app'], 'file_name': stream['file_name'], 'id':id,
                                    'stream_num': stream['stream_num'], 'frame_num': stream['frame_num'],
                                    'size': stream['size'], 'cipher_suite': stream['cipher_suite'],
                                    'message_sequence': stream['message_sequence'],
                                    'format_message_sequence': stream['format_message_sequence'],
                                    'packet_sequence': stream['packet_sequence'],
                                    'raw_packet_sequence': stream['raw_packet_sequence'],
                                    'dns': dns, 'sni': stream['sni'],
                                    'common_name': stream['common_name'], 'org_name': stream['org_name'],
                                    'payload_packet_num': stream['payload_packet_num'],
                                    'client_hello_packet_num': stream['client_hello_packet_num'],
                                    'server_hello_packet_num': stream['server_hello_packet_num'],
                                    'cert_packet_num': stream['cert_packet_num']
                                }
        # save dataset
        os.makedirs(dataset_cache_dir, exist_ok=True)
        with open(dataset_cache_path, 'wb') as fp:
            pickle.dump(datasets, fp)
    return datasets


def gen_weight_dict(data_list, params, app_list):
    # make weight dict
    packet_limit = params['packet_limit']
    weight_dict = {'dns': dict(), 'common_name': dict(), 'org_name': dict(), 'sni': dict(), 'cipher_suite': dict()}
    for item in data_list:
        app = item['app']
        size = item['size']
        frame_num = item['frame_num']
        dns = item['dns']
        common_name = item['common_name']
        org_name = item['org_name']
        sni = item['sni']
        cipher_suite = item['cipher_suite']

        # packet limit
        client_hello_packet_num = item['client_hello_packet_num']
        server_hello_packet_num = item['server_hello_packet_num']
        cert_packet_num = item['cert_packet_num']

        if dns:
            if dns not in weight_dict['dns']:
                weight_dict['dns'][dns] = len(app_list) * [0]
            if params['ref_weight'] == 'stream':
                weight_dict['dns'][dns][app_list.index(app)] += 1
            elif params['ref_weight'] == 'packet':
                weight_dict['dns'][dns][app_list.index(app)] += frame_num
            elif params['ref_weight'] == 'traffic':
                weight_dict['dns'][dns][app_list.index(app)] += size

        if common_name:
            if packet_limit == 0 or packet_limit >= cert_packet_num:
                if common_name not in weight_dict['common_name']:
                    weight_dict['common_name'][common_name] = len(app_list) * [0]
                if params['ref_weight'] == 'stream':
                    weight_dict['common_name'][common_name][app_list.index(app)] += 1
                elif params['ref_weight'] == 'packet':
                    weight_dict['common_name'][common_name][app_list.index(app)] += frame_num
                elif params['ref_weight'] == 'traffic':
                    weight_dict['common_name'][common_name][app_list.index(app)] += size

        if org_name:
            if packet_limit == 0 or packet_limit >= cert_packet_num:
                if org_name not in weight_dict['org_name']:
                    weight_dict['org_name'][org_name] = len(app_list) * [0]
                if params['ref_weight'] == 'stream':
                    weight_dict['org_name'][org_name][app_list.index(app)] += 1
                elif params['ref_weight'] == 'packet':
                    weight_dict['org_name'][org_name][app_list.index(app)] += frame_num
                elif params['ref_weight'] == 'traffic':
                    weight_dict['org_name'][org_name][app_list.index(app)] += size

        if sni:
            if packet_limit == 0 or packet_limit >= client_hello_packet_num:
                if sni not in weight_dict['sni']:
                    weight_dict['sni'][sni] = len(app_list) * [0]
                if params['ref_weight'] == 'stream':
                    weight_dict['sni'][sni][app_list.index(app)] += 1
                elif params['ref_weight'] == 'packet':
                    weight_dict['sni'][sni][app_list.index(app)] += frame_num
                elif params['ref_weight'] == 'traffic':
                    weight_dict['sni'][sni][app_list.index(app)] += size

        if cipher_suite:
            if packet_limit == 0 or packet_limit >= server_hello_packet_num:
                if cipher_suite not in weight_dict['cipher_suite']:
                    weight_dict['cipher_suite'][cipher_suite] = len(app_list) * [0]
                if params['ref_weight'] == 'stream':
                    weight_dict['cipher_suite'][cipher_suite][app_list.index(app)] += 1
                elif params['ref_weight'] == 'packet':
                    weight_dict['cipher_suite'][cipher_suite][app_list.index(app)] += frame_num
                elif params['ref_weight'] == 'traffic':
                    weight_dict['cipher_suite'][cipher_suite][app_list.index(app)] += size

    return weight_dict


def gen_data_vector(weight_dict, data_list, params, app_list):
    packet_limit = params['packet_limit']
    data_vectors = []
    for item in data_list:
        app = item['app']
        id = item['id']
        dns = item['dns']

        client_hello_packet_num = item['client_hello_packet_num']
        server_hello_packet_num = item['server_hello_packet_num']
        cert_packet_num = item['cert_packet_num']

        # packet limit
        if cert_packet_num is None:
            common_name = item['common_name']
            org_name = item['org_name']
        elif packet_limit == 0 or packet_limit >= cert_packet_num:
            common_name = item['common_name']
            org_name = item['org_name']
        else:
            common_name = None
            org_name = None

        if client_hello_packet_num is None:
            sni = item['sni']
        elif packet_limit == 0 or packet_limit >= client_hello_packet_num:
            sni = item['sni']
        else:
            sni = None

        if server_hello_packet_num is None:
            cipher_suite = item['cipher_suite']
        elif packet_limit == 0 or packet_limit >= server_hello_packet_num:
            cipher_suite = item['cipher_suite']
        else:
            cipher_suite = None

        if app in app_list:
            item_vec = [id, app_list.index(app)]
        else:
            item_vec = [id, len(app_list)]

        if params['dns']:
            if dns:
                if dns in weight_dict['dns']:
                    item_vec.extend(weight_dict['dns'][dns])
                else:
                    item_vec.extend(len(app_list) * [0])
            else:
                item_vec.extend(len(app_list) * [0])

        if params['common_name']:
            if common_name:
                if common_name in weight_dict['common_name']:
                    item_vec.extend(weight_dict['common_name'][common_name])
                else:
                    item_vec.extend(len(app_list) * [0])
            else:
                item_vec.extend(len(app_list) * [0])

        if params['org_name']:
            if org_name:
                if org_name in weight_dict['org_name']:
                    item_vec.extend(weight_dict['org_name'][org_name])
                else:
                    item_vec.extend(len(app_list) * [0])
            else:
                item_vec.extend(len(app_list) * [0])

        if params['sni']:
            if sni:
                if sni in weight_dict['sni']:
                    item_vec.extend(weight_dict['sni'][sni])
                else:
                    item_vec.extend(len(app_list) * [0])
            else:
                item_vec.extend(len(app_list) * [0])

        if params['cipher_suite']:
            if cipher_suite:
                if cipher_suite in weight_dict['cipher_suite']:
                    item_vec.extend(weight_dict['cipher_suite'][cipher_suite])
                else:
                    item_vec.extend(len(app_list) * [0])
            else:
                item_vec.extend(len(app_list) * [0])
        # TODO: add packet limit for packet and message
        length_count = params['length_count']
        if params['length_sequence'] == 'packet':
            if packet_limit == 0:
                packet_sequence = item['packet_sequence']
            else:
                packet_sequence = item['packet_sequence'][:packet_limit]

            if params['window_size']:
                packet_window_lengths = []
                for packet in packet_sequence:
                    packet_window_lengths.append(int(packet['payload_len']))
                    packet_window_lengths.append(int(packet['window_size']))
                length_vec = packet_window_lengths[:2*length_count] + [0] * (2*length_count - len(packet_window_lengths))
            else:
                packet_lengths = [int(packet['payload_len']) for packet in packet_sequence]
                length_vec = packet_lengths[:length_count] + [0] * (length_count - len(packet_lengths))
        elif params['length_sequence'] == 'message':
            limited_messages = []
            for message in item['message_sequence'][:length_count]:
                if message[3] is None:
                    limited_messages.append(message)
                elif packet_limit == 0 or packet_limit >= message[3]:
                    limited_messages.append(message)

            message_lengths = [message[1] for message in limited_messages]
            length_vec = message_lengths[:length_count] + [0] * (length_count - len(message_lengths))
        elif params['length_sequence'] == 'format_message':
            # packet limit for format message
            # message_lengths = []
            # for message in item['format_message_sequence']:
            #     if message[3] is None:
            #         message_lengths.append(message[1])
            #     elif packet_limit == 0 or packet_limit >= message[3]:
            #         message_lengths.append(message[1])
            #     else:
            #         message_lengths.append(0)
            # length_vec = message_lengths[:length_count] + [0] * (length_count - len(message_lengths))

            limited_messages = []
            for message in item['message_sequence'][:length_count]:
                if message[3] is None:
                    limited_messages.append(message)
                elif packet_limit == 0 or packet_limit >= message[3]:
                    limited_messages.append(message)

            handshake_items = {'22:0': ('22:0', 0, None, None),
                               '22:1': ('22:1', 0, None, None),
                               '22:2': ('22:2', 0, None, None),
                               '22:3': ('22:3', 0, None, None),
                               '22:4': ('22:4', 0, None, None),
                               '22:11': ('22:11', 0, None, None),
                               '22:12': ('22:12', 0, None, None),
                               '22:13': ('22:13', 0, None, None),
                               '22:14': ('22:14', 0, None, None),
                               '22:15': ('22:15', 0, None, None),
                               '22:16': ('22:16', 0, None, None),
                               '22:20': ('22:20', 0, None, None),
                               '22': ('22', 0, None, None),
                               }
            application_data_list = list()
            # divide handshake message first
            for message in limited_messages:
                message_type = message[0]
                if '22' in message_type:
                    handshake_items[message_type] = message
                if '23' in message_type:
                    application_data_list.append(message)

            ordered_messages = list(handshake_items.values()) + application_data_list

            message_lengths = [message[1] for message in ordered_messages]
            format_message_count = length_count + 13
            length_vec = message_lengths[:format_message_count] + [0] * (format_message_count - len(message_lengths))

        item_vec.extend(length_vec)
        data_vectors.append(item_vec)

    return data_vectors

