import sys,os
import numpy as np
import string
from subprocess import Popen, PIPE

def checkdir(path):
    if not os.path.exists(path): 
        os.makedirs(path)

def lprint(msg):
    sys.stdout.write('\r')
    sys.stdout.write(msg)
    sys.stdout.flush()

def evalf(f,X,msg=''):
    out=[]
    for i in range(len(X)):
        lprint('%d/%d %s'%(i+1,len(X),msg))
        out.append(f(X[i]))
    print()
    return np.array(out)

def gen_code(code,fname):
    code=[ _ for _ in  code.split(r'\n')]
    #code=[l+'\n' for l in code]
    F=open(fname,'w')
    F.writelines(code)
    F.close()
    
    
def send_command(cmd):
    cmd=cmd.split()
    proc = Popen(cmd, stdout=PIPE)
    stdout_value = proc.communicate()[0].decode("utf-8") 
    return stdout_value


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))