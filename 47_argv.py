# @Author  : YashowHoo
# @File    : 47_argv.py
# @Description :
import sys

def main():
    print(sys.argv, sys.argv[0], len(sys.argv))  # command-line argument, ['47_argv.py'] 47_argv.py 1, separated by spaces


if __name__ == '__main__':
    main()
