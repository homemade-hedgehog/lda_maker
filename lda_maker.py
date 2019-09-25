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


def init_sentence_piece(file_name_sentence_piece_model: str) -> spm.SentencePieceProcessor:
    """
    sentencepiece model loader
    :param file_name_sentence_piece_model: str, path to sentencepiece model file
    :return: sentecepiece.SentencePieceProcessor
    """
    sp = spm.SentencePieceProcessor()
    sp.load(file_name_sentence_piece_model)
    return sp


def aggregate_all_token(tokens_list: list) -> list:
    """
    tokens_listに含まれる全ての語を1つのlistにして返す
    :param tokens_list:list of list of str, 1文書をtokenizeしたlistを全文書でlist化
    :return: list, 結合したもの
    """
    all_token = []
    _ = [all_token.extend(tokens) for tokens in tokens_list]
    return all_token


def search_frequent_token(tokens_list: list, threshold_remove_doc_freq_rate_over_this: float) -> dict:
    """
    いろいろな文書に頻出する語は、文書を象徴する語にはならない
    文書出現割合の高いtokenはstop wordに設定
    てきとー、超テキトーに作った

    :param tokens_list: list of list of str, 1文書をtokenizeしたlistを全文書でlist化
    :param threshold_remove_doc_freq_rate_over_this: float(0-1), この割合以上の出現率の語をstop word入り
    :return:dict, {stop_word: True}, この辞書は頻出語をキーに持ち、値はTrue
    """
    # TODO いつかまともに考えるかも
    dict_doc_freq = {}
    dict_freq_vs_token = defaultdict(list)
    for tokens in tokens_list:
        for token in set(tokens):
            dict_doc_freq[token] = dict_doc_freq.get(token, 0) + 1
    for token, doc_freq in dict_doc_freq.items():
        dict_freq_vs_token[doc_freq].append(token)
    del dict_doc_freq

    # 上位の方は大体カウント値に対して1語が当たっていて、頻出top n rateを弾く
    num_remove = len(tokens_list) * threshold_remove_doc_freq_rate_over_this
    stops = []
    _ = [stops.extend(tokens) for doc_freq, tokens in dict_freq_vs_token.items() if doc_freq > num_remove]

    dict_is_stops = {token: True for token in stops}

    return dict_is_stops


class LdaMaker:
    """
    gensim LDAのラッパー
    """

    def __init__(self):
        self.path_to_save = "placeholder"
        self.sp = spm.SentencePieceProcessor()
        self.tokens_list = []
        self.dict_is_high_freq_token = {}
        self.num_tokens_variation = -999

    def set_save_directory(self, path_to_save: str) -> bool:
        """
        set & prepare path to save
        :param path_to_save: str, セーブ先フォルダ
        :return:
        """
        if path_to_save == "":
            path_to_save = f"lda_{hashlib.sha1(time.ctime().encode()).hexdigest()}/"
        if not path_to_save.endswith("/"):
            path_to_save += "/"
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)
        self.path_to_save = path_to_save

        return True

    def init_sentence_piece(self, file_name_sentence_piece_model: str) -> bool:
        """
        sentencepiceモデルのロード
        :param file_name_sentence_piece_model:
        :return:
        """
        self.sp = init_sentence_piece(file_name_sentence_piece_model)

        return True

    def make_tokens_list(self, sentences: list, threshold_remove_doc_freq_rate_over_this):
        """
        文書群を文書ごとにtokenize

        init_sentence_piece実行後に実施のこと
        :param sentences: list of str, 文書群
        :param threshold_remove_doc_freq_rate_over_this: float(0-1), この割合以上の出現率の語をstop word入り
        :return:
        """
        self.tokens_list = [self.sp.EncodeAsPieces(sentence) for sentence in tqdm(sentences, desc="tokenize @ lda")]
        # remove high doc_freq tokens
        # ## make frequently token dict
        self.dict_is_high_freq_token = search_frequent_token(tokens_list=self.tokens_list,
                                                             threshold_remove_doc_freq_rate_over_this=threshold_remove_doc_freq_rate_over_this)
        # ## remove
        self.tokens_list = [remove_stops(tokens, self.dict_is_high_freq_token) for tokens in self.tokens_list]
        # 異なり語数を計算しておく, ldaの再学習で使う
        self.num_tokens_variation = len(set(aggregate_all_token(self.tokens_list)))

        return True
