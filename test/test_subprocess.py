import subprocess

proc = subprocess.Popen(['python', 'test/sub.py'])
proc.wait()