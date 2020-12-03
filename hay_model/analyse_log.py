
import datetime


def parse_task(line):
    
    line = line.rstrip()
    line = line.replace('\t', ' ')
    line = line.split(' ')
    line = [l for l in line if l]
    
    node = line[-1]
    day = line[0].split("-")
    time = line[1].split(":")
    second = time[2].split('.')[0]
    microsecond = time[2].split('.')[1]

    date = datetime.datetime(
        year=int(day[0]),
        month=int(day[1]),
        day=int(day[2]),
        hour=int(time[0]),
        minute=int(time[1]),
        second=int(second),
        microsecond=int(microsecond)
    )

    if 'finished' in line:
        status = 'finish'
    elif 'arrived' in line:
        status = 'start'
    else:
        raise Exception('Not finished nor arrived !')

    return {'node': node, 'date': date, 'status': status}


def parse_generation(line):
    
    line = line.rstrip()
    line = line.replace('\t', ' ')
    line = line.replace("INFO:__main__:", "")
    line = line.split(' ')
    line = [l for l in line if l]
    
    if len(line) == 6:
        return {
            'ngen': line[0],
            'offspring_size': line[1],
            'avg': line[2],
            'std': line[3],
            'min': line[4],
            'max': line[5],
        }

    else:
        #raise Exception('Incorrect INFO line')
        print('Incorrect INFO line')

        
def analyse_log(filepath):

    with open(filepath, 'r') as fp:
        lines = fp.readlines()

    tasks = []
    generations = [{'tasks': [], 'stats': {}}]
    skip_one = True
    for line in lines:

        if 'task::task' in line:
            tsk = parse_task(line[:])

            # Find the start of the process and compute the dt
            if tsk['status'] == 'finish':
                for task in tasks:
                    if task['status'] == 'start':
                        if task['node'] == tsk['node']:
                            generations[-1]['tasks'].append(
                                (tsk['date']-task['date']).total_seconds()
                            )

            else:
                tasks.append(tsk)

        if 'INFO:__main__:' in line and not 'checkpoint' in line and not 'Generation' in line:
            
            if skip_one:
                skip_one=False
                continue

            # End of generation
            generations[-1]['stats'] = parse_generation(line[:])
            
            if int(len(generations[-1]['tasks'])) != int(generations[-1]['stats']['offspring_size']):
                print('Missing {} tasks for ngen {}'.format(
                    int(generations[-1]['stats']['offspring_size'])-int(len(generations[-1]['tasks'])), 
                    generations[-1]['stats']['ngen'])
                     )
                
            generations.append({'tasks': [], 'stats': {}})
            tasks = []

    return generations
