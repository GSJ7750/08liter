#!/usr/bin/env python
# coding: utf-8


import requests
from bs4 import BeautifulSoup
import regex as re
import json
from tqdm import tqdm
import time
import pandas as pd
import os

from utils.mysql import MySQL, get_db_info


def sub(x):
    return re.sub(' ','',x)


class Category_Crawler:
    def __init__(self):
        self.cat_length = 9
        self.cat_dict = dict()
        self.web_category = [[['없음','없음','없음','없음','없음']]]*self.cat_length
        self.flat_web_category = None
        self.cat_names = list()
        
        self.db = None
        self.open_db()
        self.db_category = None #list(dict()) 형태
        
        self.db_cat_df = None
        self.web_cat_df = None
        self.outer_df = None
        
        
    
    def get_category_from_json(self, category_code):
        url = 'https://search.shopping.naver.com/search/category?catId={}'.format(category_code)
        html = requests.get(url)
        time.sleep(0.1)
        html = html.text
        soup = BeautifulSoup(html, 'html.parser')

        script = soup.find("script", {"id":"__NEXT_DATA__"})
        category_dict = json.loads(script.next)['props']['pageProps']['initialState']['mainFilters'][0]['filterValues']

        result = [[category['title'], category['value']] for category in category_dict]
        return result
    
    
    def get_third_tier_with_subgroup(self, first_tier_id):#카테고리 테이블 페이지
        result = dict()
        cat_names_from_whole_category = list()

        url = 'https://search.shopping.naver.com/category/category.nhn?cat_id={}'.format(first_tier_id)
        html = requests.get(url)
        time.sleep(0.1)
        html = html.text
        
        soup = BeautifulSoup(html, 'html.parser')
        
        
        first_regex = re.compile('.category_tit.')
        first_tier_name = soup.find("h2", {"class" : first_regex}).text.strip()
        cat_names_from_whole_category.append([first_tier_name, first_tier_id])
        

        second_regex = re.compile('.category_cell.')
        second_tiers = soup.find_all('div', {'class' : second_regex})
        for second_tier in second_tiers:

            second_tier_name = re.sub(' ','',second_tier.find('h3').find('strong').text.strip())
            second_tier_id = second_tier.find('h3').find('a')['href'].split('catId=')[-1]
            cat_names_from_whole_category.append([second_tier_name, second_tier_id])

            if second_tier_name == 'PC주변기기': #카테고리 테이블과 제품검색 페이지 카테고리명이 다른경우
                second_tier_name = '주변기기'
            result[second_tier_name] = dict()
            
            if second_tier_name in ['노트북', '태블릿PC', '모니터', '마라톤용품', '당구용품', '기타스포츠용품', '제화브랜드']:   # 소분류 없는 중분류들
                continue

            second_list_regex = re.compile('.category_list.')
            for i, child in enumerate(second_tier.find('ul', {'class' : second_list_regex}).children):
                if child.find('a') != -1:
                    third_tier = child
                    third_tier_name = re.sub(' ','',child.find('a').text.strip())
                    third_tier_id = child.find('a')['href'].split('catId=')[-1]
                    cat_names_from_whole_category.append([third_tier_name, third_tier_id])

                    if third_tier.find('ul'):
                        result[second_tier_name][third_tier_name] = {'id':third_tier_id}
                        
                        for fourth_tier_1 in third_tier.find_all('ul'):     #more_on과 일반적인 리스트가 동시에 있는 경우가 있음
                            for fourth_tier in fourth_tier_1.find_all('li'):#more_on과 일반적인 리스트가 동시에 있는 경우가 있음
                                fourth_tier_name = fourth_tier.text.strip()
                                fourth_tier_id = fourth_tier.find('a')['href'].split('catId=')[-1]
                                cat_names_from_whole_category.append([fourth_tier_name, fourth_tier_id])
                        
        for k in list(result.items()):
            if len(k[1]) == 0 :
                result.pop(k[0])
                
        self.cat_names.extend(cat_names_from_whole_category)
        return result
    
    
    def get_category_table(self, first_tier_id):#상품 검색 페이지
        result = list()
        url = 'https://search.shopping.naver.com/search/category?catId={}'.format(first_tier_id)
        html = requests.get(url)
        time.sleep(0.1)
        html = html.text
        soup = BeautifulSoup(html, 'html.parser')

        first_regex = re.compile('.category_info.')
        first_tier_name = soup.find("div", {"class":first_regex}).text.strip()
        result.append([first_tier_name, None, None, None, first_tier_id])

        subgroup_dict = self.get_third_tier_with_subgroup(first_tier_id)

        second_tier = self.get_category_from_json(first_tier_id)
        for second in tqdm(second_tier, desc=' id:{}, name:{}    '.format(first_tier_id, first_tier_name)):
            second[0] = re.sub(' ','',second[0])
            result.append([first_tier_name, second[0], None, None, second[1]])
            if second[0] in ['노트북', '태블릿PC', '모니터', '마라톤용품', '당구용품', '기타스포츠용품', '제화브랜드']:   # 소분류 없는 중분류들
                continue

            third_tier = self.get_category_from_json(second[1])
            for third in third_tier:
                third[0] = re.sub(' ','',third[0])
                result.append([first_tier_name, second[0], third[0], None, third[1]])

                if second[0] in subgroup_dict.keys():
                    if third[0] in subgroup_dict[second[0]]:
                        fourth_tier = self.get_category_from_json(subgroup_dict[second[0]][third[0]]['id'])
                        for fourth in fourth_tier:
                            result.append([first_tier_name, second[0], third[0], fourth[0], fourth[1]])
        return result
    
    def update_cat_dict(self, start_code, end_code):
        code_list = [i for i in range(start_code, end_code)]
        while(len(code_list) > 0):
            try:
                for i in range(code_list[0], code_list[-1]+1):
                    category_table = self.get_category_table('{}'.format(i))
                    self.web_category[i-50000000] = category_table
                    for data in category_table:
                        if data[0] and data[1] == None:
                            self.cat_dict[data[0]] = {'name' : data[0], 'id':data[4], 'parent':None}

                        if data[1] and data[2] == None:
                            self.cat_dict[data[0]][data[1]] = {'name' : data[1], 'id':data[4], 'parent':data[0]}

                        if data[2] and data[3] == None:
                            self.cat_dict[data[0]][data[1]][data[2]] = {'name':data[2], 'id':data[4], 'parent':data[1]}

                        if data[3]:
                            self.cat_dict[data[0]][data[1]][data[2]][data[3]] = {'name':data[3], 'id':data[4], 'parent':data[2]}

                    code_list.remove(i)
            
            except Exception as e:
                print('Connection Failed, Retry...', e)
        self.get_flat_web_cat()
        
    def get_flat_web_cat(self):
        self.flat_web_category = [y for x in self.web_category for y in x]#flatten
        
    def save_cat_dict_to_json(self, path = 'category_crawling/category.json'):
        with open(path, 'w') as f:
            json.dump(self.cat_dict, f)
    
    
    def open_db(self):
        host, db_name, user, password = get_db_info()
        self.db = MySQL(host, db_name, user, password)
        
    def get_db_category_table(self):
        self.db_category = self.db.execute('select sc.id, sc.category_code, cl.name                                     from zeliterai.site_category sc                                     join zeliterai.category_language cl on sc.category_id = cl.category_id                                     where sc.site_id = 3 order by sc.category_code')
        
    def get_category_dataframe(self):
        self.get_db_category_table()
        self.db_cat_df = pd.DataFrame([[d['name'], d['category_code']] for d in self.db_category], columns=['name', 'id'])
        self.web_cat_df = pd.DataFrame(self.flat_web_category, columns=['1','2','3','4','id'])
        
    def get_duplicted_web_category(self):
        return self.web_cat_df[self.web_cat_df['id'].duplicated(keep=False)].sort_values(['id'])
    
    def find_category_from_web(self, category_code):
        return self.web_cat_df[self.web_cat_df['id'] == '{}'.format(category_code)]
    
    def find_category_from_db(self, category_code):
        return self.db_cat_df[self.db_cat_df['id'] == '{}'.format(category_code)]
    
    def get_web_db_outer_join(self):
        flat_cat_list = list()
        for c in self.flat_web_category:
            if c[3] != None:
                flat_cat_list.append([c[3],c[4]])
            if c[2] and c[3] == None:
                flat_cat_list.append([c[2],c[4]])
            if c[1] and c[2] == None:
                flat_cat_list.append([c[1],c[4]])
            if c[0] and c[1] == None:
                flat_cat_list.append([c[0],c[4]])
                
        flat_web_cat_df = pd.DataFrame(flat_cat_list, columns=['name', 'id'])
        
        self.outer_df = pd.merge(flat_web_cat_df, self.db_cat_df, left_on='id', right_on='id', how='outer')
        self.outer_df.columns = ['naver', 'category_code', 'db']
        self.outer_df = self.outer_df[self.outer_df['naver'].isna() | self.outer_df['db'].isna()].fillna('없음')
        return self.outer_df
    
    def save_outer_df_to_csv(self, path='category_crawling/outer.csv'):
        self.outer_df.to_csv(path, index_label=False, index=False)
        
        
        
    def validate(self, path = 'category_crawling/참고.csv'):
        tmp_list = list()
        cat_df = pd.DataFrame(self.cat_names, columns=['name', 'id'])
        ids = cat_df['id'].values.tolist()
        
        for t in self.web_cat_df['id'].values.tolist():
            if (t not in ids) and t != '없음':
                tmp_list.append(t)
                
        result = sorted(list(set(tmp_list)))
        self.web_cat_df[self.web_cat_df['id'].isin(result)].to_csv(path,index=False)
        return result


if __name__ == '__main__':
    cc = Category_Crawler()
    cc.update_cat_dict(50000000, 50000009)
    
    directory = os.path.join('category_crawling')
    if not os.path.isdir(directory):
        os.mkdir(directory)
    
    cc.save_cat_dict_to_json()
    cc.get_category_dataframe()
    
    cc.get_web_db_outer_join()
    cc.save_outer_df_to_csv()
    cc.validate()





