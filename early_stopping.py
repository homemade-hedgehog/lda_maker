from gensim.models import LdaMulticore
import random
import numpy as np


def split_train_valid(corpus: list, rate_of_valid: float) -> (list, list):
    """
    train, validを分割して返す
    :param corpus: list{iterable of (int, int or float)}, 学習予定のコーパス
    :param rate_of_valid: float, 0 < rate_of_valid < 1, validationに回すデータの割合
    :return:
    """
    if rate_of_valid <= 0:
        rate_of_valid = 0.01
    elif rate_of_valid >= 1:
        rate_of_valid = 0.99
    len_corpus = len(corpus)
    len_valid = int(len_corpus * rate_of_valid)
    index_valid = random.sample(list(range(len_corpus)), len_valid)
    return [corpus[i] for i in set(range(len_corpus)).difference(index_valid)], [corpus[i] for i in index_valid]


class EarlyStopping:
    def __init__(self, patience, must_move_this_rate):
        self.log = []
        self.patience = patience
        self.must_move_this_rate = must_move_this_rate

    def check_perplexity(self, model: LdaMulticore, valid_corpus: list) -> bool:
        """
        perwordbound から perplexityを計算してログに追加
        - perplexity: 要はエントロピー。詳しくは書籍:統計的学習の基礎(厚み10cm!)が説明が端折られず分かりやすい
        - perwordbound:　
          - gensim標準のperplexity計算メソッドlog_perplexityの戻り値
          - the variational bound of documents from the corpus as E_q[log p(corpus)] - E_q[log q(corpus)]
          - 推定分布と真の分布の差...迷いの大きさに-1をかけた値と思っておけばOK

        :param model: LdaMulticore gensimのLDAモデル
        :param valid_corpus: list{iteratable of (int, int or float)}, コーパス
        :return:
        """
        perwordbound = model.log_perplexity(valid_corpus)
        perplexity = np.exp2(-perwordbound)
        self.log.append(perplexity)

        return True

    def is_converged(self, model: LdaMulticore, valid_corpus: list) -> bool:
        """
        収束判定をして結果を返す
        True: 収束
        False: 収束してない
        :param model: LdaMulticore gensimのLDAモデル
        :param valid_corpus: list{iteratable of (int, int or float)}, コーパス
        :return:
        """
        self.check_perplexity(model=model, valid_corpus=valid_corpus)
        if len(self.log) > max(2, self.patience):
            diff = np.diff(np.array(self.log))
            # 直近patience回の変化率平均が小さいなら、多分収束
            flag_limit_of_patience = np.abs(1 - np.mean(diff[-self.patience])) < self.must_move_this_rate

            # 初期変化量のmust_move_this_rate(float < 1)も変わらなければ、多分収束
            flag_not_moving = np.abs(diff[-1] / diff[0]) < self.must_move_this_rate
            if flag_limit_of_patience or flag_not_moving:
                return True
            return False
        else:
            return False
