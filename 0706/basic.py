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
        s = value.get('s')
        l = value.get('l')
        
        print(h, t, s, l)


if __name__ == '__main__':
    main()
