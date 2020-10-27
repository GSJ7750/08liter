import pandas as pd
import pickle, re
import sentencepiece as spm
from konlpy.tag import Mecab
from konlpy.tag import Okt
from tqdm import tqdm
from soynlp.normalizer import repeat_normalize
from gensim.utils import simple_preprocess
from gensim.models import Phrases
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from collections import Counter
import pprint

class ReviewLDA():
    def __init__(self):
        self.__tokenizer_type = None
        self.dataset = None
        self.context = []
        self.vocab_dict = None
        self.lda_model = None
        self.regex = re.compile('[^ 가-힣]+')
    
    def __check_dataset(self, dataset):
        if type(dataset) != pd.core.frame.DataFrame:
            raise TypeError('데이터 형식을 확인하십시오 (pandas dataframe)')
        if '리뷰' not in dataset.columns:
            raise ValueError('데이터에 "리뷰" column이 없습니다.')

    def load_pkl(self, dataset_path):
        """Pickle 형식의 데이터를 읽습니다."""
        with open(dataset_path, 'rb') as f:
            dataset = pickle.load(f)
        self.__check_dataset(dataset)
        self.dataset = dataset['리뷰'].values
    
    def load_excel(self, dataset_path):
        dataset = pd.read_excel(dataset_path)
        self.__check_dataset(dataset)
        self.dataset = dataset['리뷰'].values
    
    def load_csv(self, dataset_path):
        dataset = pd.read_csv(dataset_path)
        self.__check_dataset(dataset)
        self.dataset = dataset['리뷰'].values
    
    def load_tokenizer(self, method, spm_path=None):
        """method(str) : spm, mecab
           spm_path(str) : method가 spm이면 spm 모델 경로 입력"""
        if method not in ['spm', 'mecab', 'okt']:
            raise ValueError('잘못된 method 입력됨')
        self.__tokenizer_type = method
        if method == 'spm':
            if not spm_path:
                raise ValueError('spm_path가 존재하지 않습니다.')
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.load(spm_path)
        elif method == 'okt':
            self.tokenizer = Okt()
        else:
            self.tokenizer = Mecab()

    def __tokenize(self, text):
        text = self.regex.sub('', text)
        if not self.__tokenizer_type:
            raise ValueError('Tokenizer를 먼저 load하세요')
        if self.__tokenizer_type == 'spm':
            return [repeat_normalize(token.replace("▁", "")) for token in self.tokenizer.EncodeAsPieces(text)]
        elif self.__tokenizer_type == 'okt':
            tag_list = ['Noun', 'Verb', 'Adjective', 'Adverb']
            return [repeat_normalize(token) for token, pos in self.tokenizer.pos(text) if pos in tag_list]
        else:
            return [repeat_normalize(token) for token in self.tokenizer.nouns(text)]

    def kor_preprocess(self):
        for review in tqdm(self.dataset):
            self.context.append(self.__tokenize(review))
    
    def make_ngram(self, n, min_count=10):
        result = list()
        bigram = Phrases(self.context, min_count=min_count, threshold=10)
        if n == 2:
            for doc in tqdm(self.context):
                result.append(bigram[doc])
        elif n ==3 :
            trigram = Phrases(bigram[self.context])
            for doc in tqdm(self.context):
                result.append(trigram[doc])
        else:
            raise ValueError('n그램 값이 너무 큽니다.(2 혹은 3)')
        self.context = result
    
    def make_vocab(self):
        count_dict = Counter()
        for review in tqdm(self.context):
            count_dict.update(Counter(review))
        self.vocab_dict = dict(sorted(count_dict.items(), key=(lambda x:x[1]), reverse=True))

    def get_vocab(self, reverse=True):
        if reverse:
            return self.vocab_dict
        else:
            return dict(sorted(self.vocab_dict.items(), key=(lambda x:x[1]), reverse=False))
            
    
    def filter_vocab(self, min_count=0, max_count=None):
        """min_count (int) : 이 이상 등장한 단어만 사용
           max_count (int) : 이 이하 등장한 단어만 사용"""
        if not self.vocab_dict:
            raise ValueError("vocab을 먼저 생성해주십시요 (make_vocab method)")
        if not max_count:
            max_count = len(self.context)
        self.vocab_dict = dict([(k, v) for k, v in self.vocab_dict.items() if v > min_count if v < max_count])
    
    def do_lda(self, num_topics, workers=8, iterations=400, passes=15):
        self.id2word = dict([(i, k) for i, k in enumerate(self.vocab_dict.keys())])
        self.word2id = dict([(k, i) for i, k in enumerate(self.vocab_dict.keys())])
        self.corpus = list()
        
        for review in tqdm(self.context):
            self.corpus.append(self.__get_doc2bow(review))
            
        print('Fitting Start.')
        self.lda_model = LdaMulticore(corpus=self.corpus, num_topics=num_topics,id2word=Dictionary().from_corpus(self.corpus, self.id2word),
                                workers=workers, iterations=iterations, passes=passes)
        print('Model Fitted.')

    def print_lda(self):
        if not self.lda_model:
            raise ValueError('모델을 먼저 학습하십시오')
        pprint.pprint(self.lda_model.print_topics())

    
    def __get_doc2bow(self, review):
        counter = dict()
        for token in review:
            if self.word2id.get(token):
                counter[self.word2id[token]] = counter.get(self.word2id[token], 1) + 1
        return list(counter.items())

if __name__ == '__main__':
    None