import sentencepiece as spm
from gensim.models import LdaMulticore
from gensim.corpora.dictionary import Dictionary
from tqdm import tqdm
import numpy as np
from collections import Counter, defaultdict
import gc
import os
import logging
import hashlib
import time
import joblib
import re


def remove_stops(tokens: list, dict_is_stops: dict) -> list:
    """
    stop wordをlist of token から除去する
    stop word(有ってもなくても文意が変わらない or あらゆる文書に現れるため特徴にならない)

    stop wordか否かをいちいちlist探索＆ifで除去していると時間がかかるので、
    pythonのdictの組み込み関数getで含めばvalue, 含まなければ指定値を返すのを使う
    :param tokens: list of str, token の　list
    :param dict_is_stops: dict, {stop_word: True}が入ったdict, stop wordしか持たない
    :return: list, stop word 除去済みtokens
    """
    list_not_stop = [not dict_is_stops.get(token, False) for token in tokens]
    return np.array(tokens, dtype=object)[list_not_stop].tolist()


def leave_valid(tokens: list, dict_is_valid: dict) -> list:
    """
    残したいtokenだけにする

    dict_is_valid = {token_valid: True}
    :param tokens: list of str, token の list
    :param dict_is_valid: dict
    :return: list
    """
    list_valid = [dict_is_valid.get(token, False) for token in tokens]
    return np.array(tokens, dtype=object)[list_valid].tolist()


def init_sentence_piece(file_name_sentence_piece_model: str):
    """
    sentencepiece model loader
    :param file_name_sentence_piece_model: str, path to sentencepiece model file
    :return: sentecepiece.SentencePieceProcessor
    """
    sp = spm.SentencePieceProcessor()
    sp.load(file_name_sentence_piece_model)
    return sp
