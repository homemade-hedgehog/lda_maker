from gensim.models.callbacks import PerplexityMetric
import numpy as np

"""
FUTURE WARNING
maybe delete
"""


class EarlyStoppingException(Exception):
    """
    if loss does not change, raise this
    """
    pass


class EarlyStoppingPerplexityMetric(PerplexityMetric):
    def __init__(self, corpus=None, logger=None, viz_env=None, title=None):
        """
        LDAは猛烈に時間がかかるのでearly stopで打ち切る
        そのうちgpuで組んで高速化できるだろうか？
        gumbel matrix trickあたりで行けそうな気もする

        パラメーターはスーパークラス参照
        :param corpus:
        :param logger:
        :param viz_env:
        :param title:
        """
        # init super
        super(EarlyStoppingPerplexityMetric, self).__init__(corpus=None, logger=None, viz_env=None, title=None)
        self.log = []
        self.patience = 10
        self.must_move_this_rate = 0.03

    def get_value(self, **kwargs):
        """
        add perplexity into log and stop
        Get the coherence score.

        Parameters
        ----------
        **kwargs
            Key word arguments to override the object's internal attributes.
            A trained topic model is expected using the 'model' key. This can be of type
            :class:`~gensim.models.ldamodel.LdaModel`, or one of its wrappers, such as
            :class:`~gensim.models.wrappers.ldamallet.LdaMallet` or
            :class:`~gensim.models.wrapper.ldavowpalwabbit.LdaVowpalWabbit`.

        Returns
        -------
        float
            The perplexity score.

        """
        super(PerplexityMetric, self).set_parameters(**kwargs)
        corpus_words = sum(cnt for document in self.corpus for _, cnt in document)
        # 単語1語あたりの、接続候補の数。少ないほどエントロピーが少なくて良くまとまっている
        perwordbound = self.model.bound(self.corpus) / corpus_words
        self.log.add(perwordbound)
        if len(self.log) > 2:
            diff = np.diff(np.array(self.log))
            # とても心苦しいが、early stoppingで止める方法がexception raiseしか思いつかなかった。
            # ## exceptionをこんなふうに使ってよかったっけ?
            if np.abs(1 - np.mean(diff[-self.patience]) / diff[-1]) < self.must_move_this_rate:
                # 直近patience回の平均が変わらないなら、多分収束
                raise EarlyStoppingException("no movement")
            if np.abs(diff[-1] / diff[0]) < self.must_move_this_rate:
                # 初期変化量のmust_move_this_rate(float < 1)も変わらなければ、多分収束
                raise EarlyStoppingException("converged")

        return np.exp2(-perwordbound)
