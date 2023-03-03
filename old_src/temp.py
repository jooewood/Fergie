#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zdx
"""
from chemspipy import ChemSpider


cs = ChemSpider('aeU3XOrf4RxihXwBgIlJYtrp79fIjLmv')
c1 = cs.get_compound(236)


c2 = cs.search('benzene')
