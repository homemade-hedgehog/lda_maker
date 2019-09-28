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
import codecs
from callbacks import EarlyStoppingPerplexityMetric


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

    def __init__(self, file_name_sentence_piece_model: str, path_to_save=""):
        self.path_to_save = "placeholder"
        self.set_save_directory(path_to_save)
        self.sp = spm.SentencePieceProcessor()
        self.init_sentence_piece(file_name_sentence_piece_model)
        self.tokens_list = []
        self.dict_is_high_freq_token = {}
        self.num_tokens_variation = -999
        self.num_process = os.cpu_count()
        self.dictionary = "placeholder"

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

    def make_lda_model(self, sentences: list, threshold_remove_doc_freq_rate_over_this: float, num_topics=20,
                       passes=200) -> bool:
        """
        LDAモデルを作成するラッパー
        LDA関連一式はjoblibを使って自動セーブ

        passesは学習のiteration回数で、少ないと精度がとても悪い
        精度の指標値を数値で表示するので、収束してなさそうならpassesを大きくして再実行

        reference:
        [数式多めでさらっと](http://acro-engineer.hatenablog.com/entry/2017/12/11/120000)
        [ガチ勢のため](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)
        :param sentences: list of str, 文書群, 文のlist
        :param threshold_remove_doc_freq_rate_over_this:　float(0-1), この割合以上の出現率の語をstop word入り
        :param num_topics:　int, 想定する話題の数, 意味はreference参照
        :param passes: int, LDAの学習回数, 多くすると基本的には良いことが多い
        :return:
        """
        # documents -> list of tokens. token must be not high freq
        self.make_tokens_list(sentences=sentences,
                              threshold_remove_doc_freq_rate_over_this=threshold_remove_doc_freq_rate_over_this)
        # gensim dictionary, 全異なり語数を端折らずに辞書化する指定。 * 指定しないと10,000語くらいで端折るはず
        self.dictionary = Dictionary(self.tokens_list, prune_at=self.num_tokens_variation)
        # tokens -> corpus
        corpus = [self.dictionary.doc2bow(tokens) for tokens in self.tokens_list]

        # prepare LDA
        check_perplexity = 5
        # make LDA
        # ## for convergence monitor
        file_name_log = "delete_me.log"
        logging.basicConfig(handlers=[logging.FileHandler(filename=file_name_log, mode="w", encoding='utf-8')],
                            format="%(asctime)s:%(levelname)s:%(message)s",
                            level=logging.INFO)
        # ## make
        model_lda = LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=self.dictionary,
                                 workers=self.num_process, passes=passes, eval_every=check_perplexity)
        # ## monitor shutdown
        logging.shutdown()

        # geisim ldaのログフォーマットでperplexityの行を探す
        p = re.compile(r"(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
        # ## 読んで抽出
        matches = [p.findall(l) for l in open(file_name_log, encoding="utf-8")]
        # ## hitしている行は長さがある
        matches = [m for m in matches if len(m) > 0]
        tuples = [t[0] for t in matches]
        # ## gensim perplexity means log_preplexity
        perplexity = [float(t[1]) for t in tuples]
        # ## per word bounds
        liklihood = [float(t[0]) for t in tuples]
        print("log perplesity, bigger is better\n", perplexity)
        print("if not converged. retry 'make_lda_model(passes=MORE_LARGE_INTEGER)'")
        os.remove(file_name_log)

        # save
        joblib.dump(self.dict_is_high_freq_token, f"{self.path_to_save}dict_is_stops.joblib")
        joblib.dump(model_lda, f"{self.path_to_save}model_LDA.joblib")
        joblib.dump(self.dictionary, f"{self.path_to_save}dictionary_LDA.joblib")

        return True
