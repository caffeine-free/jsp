#!/usr/bin/env python3
import itertools, csv, random, math, sys, copy, json
from operator import indexOf
import plotly.figure_factory as ff
import pandas as pd


def main():
    filename = sys.argv[1]
    args = sys.argv[2:]
    path = './instances.json'

    # irace config
    if 'instances' not in sys.argv[1]:
        filename = sys.argv[4].split('jsp/')[-1]
        args = sys.argv[5:]
        path = '../instances.json'

    parms = read_args(args)

    f = open(path)
    data = json.load(f)
    optimum = list(filter(lambda d: d['path'] == filename, data))[0]

    if optimum['optimum']: optimum = optimum['optimum']
    elif optimum['bounds']: optimum = optimum['bounds']['lower']
    else: optimum = None

    global machine_size
    global job_size

    filename = f'../{filename}' if 'instances' not in sys.argv[1] else filename
    job_matrix = get_job_matrix(filename)
    job_size, machine_size = job_matrix.pop(0)
    
    s = build_initial_solution(job_matrix)

    if parms['plot'] and job_size <= 10: plot_gantt(s)
    t0 = initial_temp(s, parms['beta'], parms['gama'], parms['samax'], parms['t0'])
    makespan = sa(s, parms['alpha'], t0, parms['samax'])
    if parms['plot'] and job_size <= 10: plot_gantt(s)

    gap = 100 *((makespan - optimum) / optimum) if optimum else None

    print(makespan, gap)

    
def sa(s, alpha, temperature, sa_max):
    s_prime = copy.deepcopy(s)
    eps = 1e-6
    
    fo = fo_n = fo_prime = get_makespan(s_prime)

    iters = 0
    while temperature > eps and iters < sa_max:
        i, j, r_machine, fo_n = gen_move(s)
        delta = fo_n - fo;

        if delta < 0:
            fo = fo_n
            do_move(s, i, j, r_machine)

            iters = 0

            if (fo_n < fo_prime):
                fo_prime = fo_n
                s_prime = copy.deepcopy(s)

        else:
            x = random.uniform(0, 1)
            if x < math.exp(-delta / temperature):
                do_move(s, i, j, r_machine)
                fo = fo_n

        iters += 1
        temperature *= alpha

    s = copy.deepcopy(s_prime)
    return fo_prime


def initial_temp(s, beta, gama, sa_max, temperature):
    fo = fo_n = get_makespan(s);

    while True:
        accepted = 0
        for _ in range(int(sa_max)):
            fo_n = gen_move(s)[3]
            delta = fo_n - fo;

            if delta < 0:
                accepted += 1
            else:
                x = random.uniform(0, 1)
                if x < math.exp(-delta / temperature):
                    accepted += 1

        if accepted >= gama * sa_max: break
        else: temperature *= beta;

    return temperature;


def gen_move(s):
    move_type = random.choice(['random, greed'])

    r_machine = random.randrange(machine_size)
    i, j = random.sample(range(job_size), 2)
    j = random.choice([j, job_size - 1])
   
    if move_type == 'greed':
        r_machine = max(enumerate(s), key=lambda item: item[1]['time'])[0]
        j = job_size - 1

    makespan = get_makespan(s)
    if j == job_size - 1:
        ms = s[r_machine]['jobs'][j - 1]['Finish']
        makespan = makespan if ms <= makespan else ms

    if is_valid_move(s, i, j, r_machine):
        return i, j, r_machine, makespan
    else:
        return j, j, r_machine, makespan


def is_valid_move(s, i, j, r_machine):
    in_job = s[r_machine]['jobs'][i]
    pi_job = s[r_machine]['jobs'][i - 1] if i != 0 else None
    
    job = s[r_machine]['jobs'][j]
    job_time = job['Finish'] - job['Start']
    job_id = s[r_machine]['jobs'][j]['Resource']

    start = s[r_machine]['jobs'][i - 1]['Finish'] if i != 0 else 0
    finish = start + job_time

    if pi_job and in_job['Start'] - pi_job['Finish'] < job_time: return False
    elif not pi_job and in_job['Start'] < job_time: return False

    for machine in s:
        for job in machine['jobs']:
            if start >= job['Start'] and start <= job['Finish']:
                if job['Resource'] == job_id: return False

            elif finish >= job['Start'] and finish <= job['Finish']:
                if job['Resource'] == job_id: return False
            
            elif start <= job['Start'] and finish >= job['Finish']:
                if job['Resource'] == job_id: return False

    return True


def do_move(s, i, j, r_machine):
    if i == j: return

    job_in = s[r_machine]['jobs'][i]

    job = s[r_machine]['jobs'].pop(j)
    job_time = job['Finish'] - job['Start']

    job['Start'] = s[r_machine]['jobs'][i - 1]['Finish'] if i != 0 else 0
    job['Finish'] = job['Start'] + job_time
    
    index = s[r_machine]['jobs'].index(job_in)
    s[r_machine]['jobs'].insert(index, job)

    if (j == job_size - 1):
        s[r_machine]['time'] = s[r_machine]['jobs'][j]['Finish']


def get_job_matrix(filename):
    job_matrix = []

    with open(filename) as f:
        if (f.readline().startswith('#')): f = itertools.islice(f, 3, None)
        else: f.seek(0)

        reader = csv.reader(f, delimiter=' ')
        for line in reader:
            line = filter(lambda name: name.strip(), line)
            it = iter([int(item) for item in line])
            job_matrix.append([ (item, next(it)) for item in it ])

    job_matrix[0] = job_matrix[0][0]
    return job_matrix


def build_initial_solution(job_matrix):
    s = [ { 'jobs': [ ], 'time': 0 } for _ in range(machine_size) ]
    t_job_matrix = transpose(job_matrix)

    for i, job_fase in enumerate(t_job_matrix):
        for j, job in enumerate(job_fase):
            prev_fase = i - 1
            machine_index, job_time = job

            curr_job = {
                'Task': str(machine_index), 
                'Start': None, 
                'Finish': None,
                'Resource': str(j)
            }

            if prev_fase == -1:
                curr_job['Start'] = s[machine_index]['time']
                s[machine_index]['time'] += job_time
                curr_job['Finish'] = s[machine_index]['time']

                s[machine_index]['jobs'].append(curr_job)
            else:
                prev_machine = t_job_matrix[prev_fase][j][0]
                s[machine_index]['jobs'].append(curr_job)
                
                if s[machine_index]['time'] < s[prev_machine]['time']:
                    s_time = get_time(s, curr_job, prev_machine, machine_index)

                    curr_job['Start'] = s_time
                    s[machine_index]['time'] = s_time + job_time
                    curr_job['Finish'] = s[machine_index]['time']
                    
                else:
                    curr_job['Start'] = s[machine_index]['time']
                    s[machine_index]['time'] += job_time
                    curr_job['Finish'] = s[machine_index]['time']

    return s


def get_time(s, curr_job, prev_machine, machine_index):
    for job in s[prev_machine]['jobs']:
        if job['Resource'] == curr_job['Resource'] \
        and job['Finish'] >= s[machine_index]['time']:
            return job['Finish']

    return s[prev_machine]['time']


def get_makespan(s):
    return max(s, key=lambda machine: machine['time'])['time']


def transpose(matrix):
    return [ *map(list, zip(*matrix)) ]


def plot_gantt(s):
    fig = ff.create_gantt(
        pd.DataFrame(itertools.chain(*[m['jobs'] for m in s])), 
        index_col='Resource', 
        show_colorbar=True, 
        group_tasks=True
    )

    fig.update_xaxes(type='linear')
    fig.show()


def read_args(args) -> None:
    parms = {
        'alpha': 0.98,
        'beta': 1.5,
        'gama': 0.98,
        'samax': 1000,
        't0': 30,
        'plot': False
    }

    index = 0
    while index < len(args):
        option = args[index]
        index += 1

        if option == '-alpha': parms['alpha'] = float(args[index])
        elif option == '-beta': parms['constructive'] = float(args[index])
        elif option == '-gama': parms['algorithm'] = float(args[index])
        elif option == '-samax': parms['samax'] = int(args[index])
        elif option == '-t0': parms['t0'] = int(args[index])
        elif option == '-plot': parms['plot'] = True
        else: print_usage()
        index += 1

    return parms


def print_usage():
    usage = \
        f'Usage: python3 src/main.py <input> [options]\n' + \
        f'    <input>  : Name of the problem input file.\n' + \
        f'\nOptions:\n' + \
        f'    -alpha    <float>\n' + \
        f'    -beta     <float>\n' + \
        f'    -gama     <float>\n' + \
        f'    -sa_max   <int>  \n' + \
        f'    -t0       <int>  \n'

    print(usage)
    sys.exit()


if __name__ == '__main__':
    main()