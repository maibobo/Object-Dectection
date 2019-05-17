#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os

path=[
"1_单行清晰船名",
"2_多行船名",
"3_待反转船名",
"4_港口名",
"5_水尺",
"6_污油箱",
"7_污水箱",
"8_严禁烟火",
"9_杜绝火种",
"10_淡水箱",
"11_POLICE",
"12_安全通道"
]

for one_path in path:

	os.makedirs(one_path)