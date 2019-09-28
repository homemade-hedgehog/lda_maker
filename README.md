# lda_maker
gensim lda wrapper

- LDAは文書群を指定された話題数になるように分類する教師なし学習の手法  
- ある文書に対して、各話題に対する当てはまり具合を0-1で推定する  
  - 0: 全く当てはまらない
  - 1: この文書は、この話題についてのみ記述されてる

- 教師なしなので、人力ラベリングなどしていない or そもそもモデルの出力結果を見てから初めて人間が評価する等の場合に用いる  
- 内部処理では、ある単語がある話題の特徴になりうるかを選り分けている  
  - 任意の文書の単語において、単語がある話題の特徴か？を集計して、文書がどの話題にどの程度当てはまるか求めている  
  - [reference](http://camberbridge.github.io/2016/07/08/%E8%87%AA%E5%B7%B1%E7%9B%B8%E4%BA%92%E6%83%85%E5%A0%B1%E9%87%8F-Pointwise-Mutual-Information-PMI-%E3%81%AB%E3%81%A4%E3%81%84%E3%81%A6/)  
  - 要は P(x,y) / P(x)P(y) = P(x|y)P(y) / P(x)P(y) = P(x|y) / P(x)　で条件yで絞ったときにxの確率が大きく動いたら特徴とみなしている  
```
例えば、食べ物の例で  
x: 鰻を食べる確率  
y: 鰻屋 or ファミレス  
鰻は1年に1回だけ食べるとする  
P(x) = 1/365  
P(x|鰻屋) = 1  
P(x|ファミレス) = 0  
P(x|鰻屋) / P(x) = 365 >> P(x|ファミレス) / P(x) = 0  
みたいなノリである話題で絞ったときに確率の比が大きく変化する条件を話題の特徴として見つけるのが基本的な考え方  
```

。。。まぁ、業界にはfasttextなどで単語->ベクトルにして、文書は単語ベクトルの単純平均にして、分類したいならkmeans++で良くない？という人もいる  
fasttextでベクトルに意味を持たせていい感じになった分と、特徴的な語を探してくれない分のバランスで勝ったり劣ったりする

### コレは何？
- gensimのLDAモデルを文書のlistと諸パラメータを入力として生成するラッパー

### usage
#### train
```python
import joblib
from lda_maker import LdaMaker
sentence_piece = r"path_to_sentence_piece/sentence_piece.model"
path_to_doc = r"path_to_document/documents.joblib"
sentences = joblib.load(path_to_doc) # ["マイクロ化学デバイスおよび解析装置\n本発明は、マイクロ化学デバイスおよび解析装置...", ...]

lda_maker = LdaMaker(sentence_piece)
lda_maker.make_lda_model(sentences, threshold_remove_doc_freq_rate_over_this=0.7, num_topics=20, passes=200, rate_of_valid=0.3, round_check_convergence=5)
```
- threshold_remove_doc_freq_rate_over_this
  - 0 ~ 1
  - doc freqがこの割合より大きい単語は評価から外す
    - doc freq: 対象の単語を含む文書数 / 総文書数
- num_topics
  - 10-100
  - 文書に含まれる話題の数
  - 選定はある程度ノリになる
  - 少ないと同じ話題に属する文書同士で話題が違うように見える。多いと違う話題に属する文書同士が同じ話題に見える
- passes
  - 20-
  - 少ないと収束まで行けずに、同じ話題に属する文書同士があからさまにおかしい感じになる
  - 多いと話題がいい感じにギュッとまとまって納得感が出てくる
- rate_of_valid
  - 0 ~ 1
  - 全文書に対し、収束評価用に取っておく文書の割合
- round_check_convergence
  - 自然数 (< passes)
  - 収束判定をround_check_convergence回に1回だけ行う
    - 収束判定は学習より処理が重いので、ある程度サボったほうが全体として軽くなる
  
#### string -> 話題の当てはまり具合
```python
import joblib
import sentencepiece as spm
from lda_maker import remove_stops
import numpy as np

# prepare sentencepiece
sp = spm.SentencePieceProcessor()
sp.load("path_to_sentence_piece/sentence_piece.model")
# load lda instance set
model = joblib.load("lda_1de19537711a9e4e19fd7dc1dc7054396d2c479f/model_LDA.joblib")
dictionary = joblib.load("lda_1de19537711a9e4e19fd7dc1dc7054396d2c479f/dictionary_LDa.joblib")
dict_is_stops = joblib.load("lda_1de19537711a9e4e19fd7dc1dc7054396d2c479f/dict_is_stops.joblib")

# target string -> token
# ## target
target_sentence = "マイクロ化学デバイスおよび解析装置\n本発明は、マイクロ化学デバイスおよび解析装置..."
# ## string -> tokens
tokens = sp.EncodeAsPieces(target_sentence)
# ## remove stop word
tokens = remove_stops(tokens, dict_is_stops)

# calculate LDA topic distribution
# ## convert token into token_id & freq
tokens_converted = dictionary.doc2bow(tokens)
# ## calculate
topic_distribution = model[tokens_converted]  # [(7, 0.9972457)] <- 話題7版に99.7%当てはまる

# topic_distribution -> vector of sentence representative
array = np.zeros(model.num_topics)
for _id, score in topic_distribution:
    array[_id] = score
# array([0.  , 0.  , 0.  , 0.  , 0.  ,0.  , 0.  , 0.99724758, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  ])

```
