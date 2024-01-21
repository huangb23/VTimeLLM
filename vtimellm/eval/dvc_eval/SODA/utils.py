#!/usr/bin/env python

def iou(interval_1, interval_2):
    """
    interval: list (2 float elements)
    """
    eps = 1e-8 # to avoid zero division
    (s_1, e_1) = interval_1
    (s_2, e_2) = interval_2

    intersection = max(0., min(e_1, e_2) - max(s_1, s_2))
    union = min(max(e_1, e_2) - min(s_1, s_2), e_1 - s_1 + e_2 - s_2)
    iou = intersection / (union + eps)
    return iou

def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])

