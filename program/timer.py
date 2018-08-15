# -*- coding: utf-8 -*-
import time

class StopWatch(object):

    def __init__(self) :
        self.make = time.time()
        return

    def start(self) :
        self.st = time.time()
        return self

    def stop(self) :
        return time.time()-self.st

    def reset(self) :
        self.st = self.make
        return self

    def __str__(self) :
        return str(self.stop())

