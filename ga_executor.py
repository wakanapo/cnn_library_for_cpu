import sys
import subprocess
from fcntl import fcntl, F_SETFL, F_GETFL
from os import O_NONBLOCK

def non_blocking_read(output):
    fd = output.fileno()
    fl = fcntl(fd, F_GETFL)
    fcntl(fd, F_SETFL, fl | O_NONBLOCK)
    try:
        return output.read()
    except:
        return ""

def run(genom_name):
    server = subprocess.Popen('python src/python/services/genom_evaluation_server.py',
                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        s_line = non_blocking_read(server.stdout)
        if s_line:
            sys.stdout.write(s_line.decode('utf-8'))
            if 'Server Ready' in s_line.decode('utf-8'):
                break
        if not s_line and server.poll() is not None:
            print('Server Error.')
            return
        
    client = subprocess.Popen('./bin/ga {}'.format(genom_name), shell=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        s_line = non_blocking_read(server.stdout)
        c_line = non_blocking_read(client.stdout)
        if s_line:
            sys.stdout.write(s_line.decode('utf-8'))
        elif server.poll() is not None:
            print('Server Error.')
            return
        if c_line:
            sys.stdout.write(c_line.decode('utf-8'))
        elif client.poll() is not None:
            server.kill()
            return

        
if __name__=='__main__':
    argv = sys.argv
    if len(argv) != 2:
        print('Usage: python ga_executor.py <genom name>')
        exit()
    run(argv[1])

