
import os
import socket
import sys

from time import time

import requests

sys.path.insert(0, '/usr/lib/python2.7/bridge/')

from bridgeclient import BridgeClient as bridgeclient
from requests.auth import HTTPBasicAuth

ID = 1
API_URL = 'http://140.113.73.142:8000/api/iot/{}/'.format(ID)


def main():
    value = bridgeclient()

    while True:
        t = value.get('t')
        h = value.get('h')
        s = value.get('s')
        l = value.get('l')
        
        payload = {
            'humidity': float(h),
            'temperataure': float(t),
            'sound':int(s),
            'luminosity':int(l),
        }
        r = requests.put(API_URL, json=payload, auth=HTTPBasicAuth('admin', 'steven8702'))
        assert r.status_code == 200

        print(h, t, s, l)


if __name__ == '__main__':
    main()
