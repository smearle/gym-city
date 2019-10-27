
def get_empty_env_ranges():
    return {'roughness':None,
          'stump_height':None,
          'obstacle_spacing':None,
          'gap_width':None}


def get_test_set_name(env_ranges):
    name = ''
    for k, v in env_ranges.items():
        if (v is not None) and (k is not 'env_param_input') and (k is not 'nb_rand_dim'):
            if k is "stump_height": # always same test set for stump height experiments
                name += k + str(v[0]) + "_3.0"
            else:
                name += k + str(v[0]) + "_" + str(v[1])
    if name == '':
        print('Empty parameter space, please choose one when launching run.py, e.g:\n'
              '"--max_stump_h 3.0 --max_obstacle_spacing 6.0" or\n'
              '"-poly" or\n'
              '"-seq"')
        raise NotImplementedError
    return name