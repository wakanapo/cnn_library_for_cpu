import sys
import subprocess

def run(genom_name):
    server = subprocess.Popen('python src/python/services/genom_evaluation_server.py',
                            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while True:
        s_line = server.stdout.readline()
        if s_line:
            sys.stdout.write(s_line.decode('utf-8'))
            if s_line.decode('utf-8') == 'ready\n':
                client = subprocess.Popen('./bin/ga {}'.format(genom_name), shell=True,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                while True:
                    c_line = client.stdout.readline()
                    if c_line:
                        sys.stdout.write(c_line.decode('utf-8'))
                    if not c_line and client.poll() is not None:
                        server.kill()
                        return
        if not s_line and server.poll() is not None:
            print('Server Error.')
            return

if __name__=='__main__':
    argv = sys.argv
    if len(argv) != 2:
        print('Usage: python ga_executor.py <genom name>')
        exit()
    run(argv[1])

