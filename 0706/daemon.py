'''
Basic function but with daemonize
'''

from __future__ import print_function

import os
import sys

sys.path.insert(0, '/usr/lib/python2.7/bridge/')

from bridgeclient import BridgeClient as bridgeclient


def main():
    value = bridgeclient()

    while True:
        t = value.get('t')
        h = value.get('h')
        print(h, t)


if __name__ == '__main__':
    main()
