# 2025.06.16
# tag_producer_v2.py
# By Tangerdream


import pandas as pd
import numpy as np
import json
import re
import random
import hashlib
import time
import uuid
import requests
import argparse
from google.genai import types
from google import genai

# # Settings类用于存储和管理程序的初始配置参数，包括预设标签、翻译服务、代理设置等。
class Settings(argparse.Namespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # #区别于英文的高级语言，用以触发关键词
        self.supper_languiage = '中文'  # 只需要对应语言，内容不限

        # #tag预设，请对应预设表格的sheetname或json的key进行设置
        self.pretag_json_path = './pretags.json' # 预设标签的JSON文件路径,若无excel则必要
        self.pretag_excel_path = './pretags.xlsx' # 预设标签的Excel文件路径,可选，若存在则优先级更高
        self.pretag_character_key_word = '人物'  # 主关键字，预设标签的关键词，对应记录角色tag或lora的sheetname或json的key
        self.pretag_character_clothes_key_word = '服装'  # 子关键字，人物表中服装的关键词,对应列名
        self.pretag_character_appearance_key_word = '外貌'  # 子关键字，人物表中外貌的关键词,对应列名
        # self.pretag_character_source_key_word = 'Source' # 人物Source的关键词,对应列名
        self.pretag_others_key_words = ['动作', '服装', '镜头', '画风', '场景', '其他'] # 主关键字，其他预设标签的关键词,对应预设表格的sheetname或json的key
        self.pretag_random_key_word = '随机' # 主关键字，随机选择的关键词

        # #绘图系统类型
        self.sys='comfyui' # 绘图系统类型，可选值为'webui'或'comfyui'

        # #有道翻译
        self.translate_Youdao = True  # 是否启用有道翻译
        self.translate_Youdao_key_word = '翻译' # 启用有道翻译的关键词，必须包含中文
        self.APP_KEY_Youdao  = '3c4cdf0cd5ebddbe'
        self.APP_SECRET_Youdao  = 'XPo7zdNw0vzBwUDQz21XJid9IFDE3vch'

        # #代理
        self.proxy = 'http://127.0.0.1:1090'   # 代理地址, None表示不使用代理

        # # Gemini
        self.GPT_Gemini = True # 是否启用Gemini
        self.GPT_Gemini_key_word = '大模型' # 启用Gemini的关键词，必须包含中文
        self.GPT_Gemini_model = "gemini-2.5-flash-preview-05-20",  # Gemini模型
        self.GPT_Gemini_api_key = "AIzaSyDNNPG_C88D8mSGM_sOte7DeTYafTasdzw"

        # # roll画师串功能
        self.roll_artist = True # 是否启用roll画师串功能
        self.roll_artist_key_word = '摇画师串'
        self.roll_artist_csv_path = './artists_cooked.csv' # roll画师串的csv文件路径，注意：csv文件必须包含“name”列

# # tag_producer类用于处理标签生成和转换，基本理念是使用中文关键字进行长英文关键字的转换。
class tag_producer():
    """
    tag_producer类用于处理标签生成和转换，基本理念是使用中文关键字进行长英文关键字的转换。
    通过excel或json文件来存储和管理标签数据。
    指令说明：
    - 人物名 (服装) (外貌) (权重)：指定人物名生成，服装和外貌可选，权重可选。人物名可查表。例：“希尔 服装 外貌 0.85”，“希尔”，“希尔 外貌 0.85”
    - （动作 动作名）/（镜头 镜头名）/（画风 画风名）/（场景 场景名）/（其他 其他名） (权重)：指定类别生成，动作、镜头、画风、场景、其他可查表，权重可选。例：“动作 睡觉”，“镜头 偷窥视角 0.85”，“画风 像素10”
    - 随机 (游戏名) (服装) (外貌) (权重): 随机从指定游戏名的角色中选择一个，服装和外貌可选，权重可选。例：“随机 绝区零 服装 外貌 0.85”，“随机 绝区零”，“随机 绝区零 外貌 0.85”
    - 随机 (动作/镜头/画风/场景/其他) (权重)：随机从指定类别中选择一个，可复选，权重可选。例：“随机 动作 动作 0.85”，“随机 镜头”，“随机 画风 镜头 0.6”
    - roll串 (整数)：根据指定的数量，从画师tag文件中随机选取对应数量画师，并生成对应长度的画师串，权重0.5~1.3之间，有0.3的概率不生成权重
    """


    def __init__(self, args):
        if args.sys not in ['webui', 'comfyui']:
            raise ValueError("sys参数必须为'webui'或'comfyui'")
        self.sys = args.sys
        self.sheetlist = args.pretag_others_key_words
        self.sheetlist.append(args.pretag_character_key_word)
        self.jobs = self.sheetlist.copy()
        self.pretag_random_key_word = args.pretag_random_key_word # 随机选择的关键词
        self.jobs.append(self.pretag_random_key_word)
        self.pretag_character_key_word = args.pretag_character_key_word  # 人物的关键词
        self.pretag_character_clothes_key_word = args.pretag_character_clothes_key_word # 人物服装的关键词
        self.pretag_character_appearance_key_word = args.pretag_character_appearance_key_word # 人物外貌的关键词

        # # 翻译
        if args.translate_Youdao:
            self.Translator= YoudaoTranslate(APP_KEY=args.APP_KEY_Youdao, APP_SECRET=args.APP_SECRET_Youdao)
            self.translate_key_word = args.translate_Youdao_key_word
            self.jobs.append(self.translate_key_word)
        else:
            self.translate_key_word = None

        # # GPT
        if args.GPT_Gemini:
            self.GPT = GeminiGPT(model=args.GPT_Gemini_model, api_key=args.GPT_Gemini_api_key)
            self.GPT_key_word = args.GPT_Gemini_key_word
            self.jobs.append(self.GPT_key_word)
        else:
            self.GPT_key_word = None

        # # 代理
        if args.proxy:
            import os
            os.environ['http_proxy'] = args.proxy
            os.environ['https_proxy'] = args.proxy

        # # roll画师串功能
        if args.roll_artist:
            self.roll_artist_key_word = args.roll_artist_key_word
            self.roll_artist = roll_artist(csv_path=args.roll_artist_csv_path)
            self.jobs.append(self.roll_artist_key_word)

        # # 预设标签文件路径
        self.jsonpath = args.pretag_json_path
        self.excelpath = args.pretag_excel_path
        if self.excelpath:
            self.dic = self.excel2dict(self.excelpath, self.sheetlist)
        elif self.jsonpath:
            with open(self.jsonpath, 'r', encoding='utf-8') as f:
                self.dic = json.load(f)
        else:
            print("请提供预设标签的Excel或JSON文件路径")

    def reset(self, args):
        self.__init__(args)
        

    def todic(self, path, key):
        dic = {}
        df = pd.read_excel(path, sheet_name=key)
        ls = df.to_dict(orient='records')
        for item in ls:
            dic[item['cname']] = item
        return dic

    def charsort(self, chardic: dict):
        charsd = {}
        for key in chardic.keys():
            if chardic[key]['Source'] not in list(charsd.keys()):
                charsd[chardic[key]['Source']] = [chardic[key]['cname']]
            else:
                charsd[chardic[key]['Source']].append(chardic[key]['cname'])
        return charsd

    def excel2dict(self, excelpath, sheetlist=None):
        if sheetlist is None:
            sheetlist = self.sheetlist
        dic = {}
        for key in sheetlist:
            dic[key] = self.todic(excelpath, key)
        dic['charsort'] = self.charsort(dic[self.pretag_character_key_word]) # 人物
        return dic

    def excel2json(self, excelpath=None, jsonpath=None, sheetlist=None):
        if sheetlist is None:
            sheetlist = self.sheetlist
        if excelpath is None:
            excelpath = self.excelpath
        if jsonpath is None:
            jsonpath = self.jsonpath
        dic = self.excel2dict(excelpath, sheetlist=sheetlist)
        # 将字典保存为 JSON 文件
        with open(jsonpath, 'w', encoding='utf-8') as f:
            json.dump(dic, f, ensure_ascii=False, indent=4)
        print('已保存为JSON文件：', jsonpath)

    def get_cn(self, prompt):
        return re.findall(r'[\u4e00-\u9fff]+[^,，]*', prompt)

    def clean_commas(self, prompt: str):
        # 使用正则表达式将连续重复的逗号替换为单个逗号
        cleaned_prompt = re.sub(r',+', ',', prompt)
        return cleaned_prompt

    def get_weights(self, aims: list):
        numlist = []
        for aim in aims:
            try:
                numlist.append(float(aim))
            except:
                pass
        if numlist:
            unet_weight = numlist[0]
            clip_weight = numlist[-1]
        else:
            unet_weight = 0
            clip_weight = 0
        return unet_weight, clip_weight

    def randkey(self, job):
        keys = self.dic[job].keys()
        return random.choice(list(keys))

    def get_tags(self, aims: list, job):
        tags = ''
        unet_weight, clip_weight = self.get_weights(aims)
        for key in aims[1:]:
            if key == self.pretag_random_key_word:
                key = self.randkey(job)
            try:
                if self.dic[job][key]['Lora']:
                    if not unet_weight:
                        unet_weight = self.dic[job][key]['unet weight']
                    if not clip_weight:
                        clip_weight = self.dic[job][key]['clip weight']

                    if self.sys == 'comfyui':
                        tags = tags + '<lora:' + self.dic[job][key]['model file name'] + ':' + str(unet_weight) + ':' + str(clip_weight) + '>,' + self.dic[job][key]['tag'] + ','
                    else:
                        # webui
                        tags = tags + '<lora:' + self.dic[job][key]['model file name'] + ':' + str(unet_weight) + '>,' + self.dic[job][key]['tag'] + ','


                else:
                    if unet_weight:
                        tags = tags + '(' + self.dic[job][key]['tag'] + ':' + str(unet_weight) + '),'
                    else:
                        tags = tags + self.dic[job][key]['tag'] + ','

            except:
                pass
        return tags

    def get_chartags(self, aims: list):
        job = self.pretag_character_key_word
        tags = ''
        if aims[0] == job:
            begin_id = 1
        else:
            begin_id = 0
        unet_weight, clip_weight = self.get_weights(aims)
        for key in aims[begin_id:]:
            if key == self.pretag_random_key_word:
                key = self.randkey(job)
            try:
                if self.dic[job][key]['Lora']:
                    if not unet_weight:
                        unet_weight = self.dic[job][key]['unet weight']
                    if not clip_weight:
                        clip_weight = self.dic[job][key]['clip weight']

                    if self.sys == 'comfyui':
                        tags = tags + '<lora:' + self.dic[job][key]['model file name'] + ':' + str(unet_weight) + ':' + str(clip_weight) + '>,' + self.dic[job][key]['name'] + ','
                    else:
                        # webui
                        tags = tags + '<lora:' + self.dic[job][key]['model file name'] + ':' + str(unet_weight)+'>,' + self.dic[job][key]['name'] + ','
                else:
                    tags = tags + self.dic[job][key]['name'] + ','

                if self.pretag_character_appearance_key_word in aims:
                    tags = tags + self.dic[job][key][self.pretag_character_appearance_key_word] + ','
                if self.pretag_character_clothes_key_word in aims:
                    tags = tags + self.dic[job][key][self.pretag_character_clothes_key_word] + ','

            except:
                pass

        return tags

    def repeatlora_del(self, input_str):
        # 使用正则表达式找出所有的lora标签
        lora_pattern = r'<lora:[^>]+>'
        lora_tags = re.finditer(lora_pattern, input_str)
        # 用于存储已经出现过的lora名称及其完整标签
        seen_loras = {}
        # 存储需要删除的标签
        to_remove = []

        # 第一遍遍历：识别重复的lora
        for match in lora_tags:
            tag = match.group()
            # 提取lora名称（第一个冒号后到第二个冒号前的内容）
            lora_name = tag[6:].split(':')[0]  # 去掉 '<lora:' 后按冒号分割

            if lora_name in seen_loras:
                # 如果这个lora已经出现过，记录需要删除的标签
                to_remove.append(tag)
            else:
                # 第一次出现的lora，记录下来
                seen_loras[lora_name] = tag

        # 第二遍：删除重复的标签
        result = input_str
        for tag in to_remove:
            result = result.replace(tag, '')
            # result = result+','+tag

        return result

    def tagrep(self, prompt: str): # 功能设置
        prompt = prompt.replace('，', ',')
        cn_list = self.get_cn(prompt)
        GPT_flag = False
        for cn in cn_list:
            aims = re.split(r'[-\s]+', cn)
            if aims[0] in self.jobs and aims[0] in self.sheetlist:
                if aims[0] == self.pretag_character_key_word: #人物
                    tags = self.get_chartags(aims)
                else:
                    tags = self.get_tags(aims, aims[0]) # 服装、动作、镜头、画风、场景、其他等等

            elif aims[0] in self.jobs and aims[0] == self.translate_key_word: # 翻译
                try:
                    aims2 = aims[1:]
                    sentence = ''
                    for i in aims2:
                        sentence = sentence + i + ','
                    tags = self.Translator.translate(sentence)
                except:
                    pass

            elif aims[0] in self.jobs and aims[0] == self.GPT_key_word: # GPT
                try:
                    GPT_flag = True
                    aims2 = aims[1:]
                    GPT_request = ''
                    for i in aims2:
                        GPT_request = GPT_request + i + ','
                except:
                    pass

            elif aims[0] in self.jobs and aims[0] == self.roll_artist_key_word: # roll画师串
                try:
                    aims2 = aims[1]
                    tags = self.roll_artist.roll_artist(int(aims2))
                except:
                    pass

            elif aims[0] == self.pretag_random_key_word: # 随机人物
                try:
                    aims2 = aims[1:]
                    aims2[0] = random.choice(self.dic['charsort'][aims2[0]])
                    tags = self.get_chartags(aims2)
                except:
                    tags = self.get_chartags(aims)
            else:
                tags = self.get_chartags(aims) # 直接指定人名

            
            prompt = prompt.replace(cn, tags)
            if GPT_flag:
                try:
                    prompt = self.GPT.refine_prompt(GPT_request, prompt)
                except:
                    pass
        return self.clean_commas(self.repeatlora_del(prompt.replace('，', ',')))
    
# # GeminiGPT类用于与Google Gemini API进行交互，主要功能是根据用户请求和原始提示词生成改进后的提示词。
class GeminiGPT():
    def __init__(self, model=None, api_key=None):
        self.model = model[0]
        self.client = genai.Client(api_key=api_key)

    def refine_prompt(self, User_Request, Original_Prompt):
        contents=f"Refine the following SDXL prompt based on the user's request ({User_Request}). Enhance or modify the descriptive elements where appropriate, but preserve the core content, especially any LoRA calls in the form <LoraName:weight>. Only return the refined prompt without additional explanation or formatting.Prompt: {Original_Prompt}"
        # print(contents)
        response = self.client.models.generate_content(
            model=self.model,
            contents=[contents],
        )
        return response.text
    
# # YoudaoTranslate类用于与有道翻译API进行交互，主要功能是添加认证参数和计算签名，以便进行翻译请求。
class YoudaoTranslate():
    def __init__(self, APP_KEY=None, APP_SECRET=None):
        self.APP_KEY = APP_KEY
        self.APP_SECRET = APP_SECRET
        self.YOUDAO_URL = 'https://openapi.youdao.com/api'

    def addAuthParams(self,appKey, appSecret, params):
        q = params.get('q')
        if q is None:
            q = params.get('img')
        q = "".join(q)
        salt = str(uuid.uuid1())
        curtime = str(int(time.time()))
        sign = self.calculateSign(appKey, appSecret, q, salt, curtime)
        params['appKey'] = appKey
        params['salt'] = salt
        params['curtime'] = curtime
        params['signType'] = 'v3'
        params['sign'] = sign


    def returnAuthMap(self,appKey, appSecret, q):
        salt = str(uuid.uuid1())
        curtime = str(int(time.time()))
        sign = self.calculateSign(appKey, appSecret, q, salt, curtime)
        params = {'appKey': appKey,
                'salt': salt,
                'curtime': curtime,
                'signType': 'v3',
                'sign': sign}
        return params

    def calculateSign(self,appKey, appSecret, q, salt, curtime):
        strSrc = appKey + self.getInput(q) + salt + curtime + appSecret
        return self.encrypt(strSrc)

    def encrypt(self,strSrc):
        hash_algorithm = hashlib.sha256()
        hash_algorithm.update(strSrc.encode('utf-8'))
        return hash_algorithm.hexdigest()

    def getInput(self,input):
        if input is None:
            return input
        inputLen = len(input)
        return input if inputLen <= 20 else input[0:10] + str(inputLen) + input[inputLen - 10:inputLen]

    def doCall(self,url, header, params, method):
        if 'get' == method:
            return requests.get(url, params)
        elif 'post' == method:
            return requests.post(url, params, header)

    def translate(self,q):
        '''
        note: 将下列变量替换为需要请求的参数
        '''
        lang_from = 'zh-CHS'
        lang_to = 'en'

        data = {'q': q, 'from': lang_from, 'to': lang_to}

        self.addAuthParams(self.APP_KEY, self.APP_SECRET, data)

        header = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = self.doCall('https://openapi.youdao.com/api', header, data, 'post')

        try:
            if response.json()["translation"][0]:
                return response.json()["translation"][0]
        except Exception as e:
            # logger.debug(f"翻译结果出错:{response.text}")
            # raise e
            pass

class roll_artist():
    def __init__(self, csv_path):
        self.artists = pd.read_csv(csv_path)
        self.length = len(self.artists['name'])

    def roll_artist(self, roll_num):
        # 产生roll_num个self.length范围内的随机整数，不重复
        roll_nums = np.random.choice(self.length, roll_num, replace=False)
        artists = ''
        for i in roll_nums:
            # 随机生成0或者1,0的概率为0.3
            if np.random.choice([0, 1], p=[0.7, 0.3]):
                artists = artists + self.artists['name'][i] + ','
            else:
                # 生成随机权重0.5~1.3，保留两位小数
                weight = np.random.uniform(0.5, 1.3)
                weight = str(round(weight, 2))
                artists = artists + '(' + self.artists['name'][i] + ':' + weight + '),'
        return artists


if __name__ == '__main__':


    args=Settings()

    tagp = tag_producer(args)
    # prompt = "noob artist:nixeu,artist:ciloranko,artist:sho_(sho_lwlw),(((nailong))), yellow skin,pokemon \(creature\) , smile,open mouth,永夜希尔 外貌 服装，standing, simple background, white background,best quality, amazing quality,shimizu,shenhe_\(frostflower_dew\)_\(genshin_impact\),1girl, blue eyes, breasts, cleavage, gradient eyes, grey hair, hair ornament, hair over one eye, long hair,"

    prompt = '摇画师串 10,随机 绝区零 服装 外貌 0.85,noob，其他 随机 随机 0.75,随机 外貌 服装 0.7，，场景-随机 0.7，画风 2d润彩 2d润彩 0.7 0.9,我啊，artist:nixeu,artist:ciloranko,artist:sho_(sho_lwlw),翻译 你好 世界 笑一个,大模型 为这个prompt设计一个沙滩元素的背景'
    print(tagp.tagrep(prompt))
    tagp.excel2json()


