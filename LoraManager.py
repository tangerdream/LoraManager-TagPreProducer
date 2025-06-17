# 2025.06.16
# LoraManager.py
# v1
# By Tangerdream

import os
import requests
import time
import hashlib
import json
import warnings
import modelscope


def get_hash(file_path):
    """使用 SHA-256 计算文件哈希值"""
    # 使用二进制模式打开文件
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # 分块读取文件以处理大文件
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def scan_folders(base_path):
    """递归扫描文件夹获取所有 .safetensors 文件"""
    safetensors_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.safetensors') or file.endswith('.ckpt'):
                safetensors_files.append(os.path.join(root, file))
    return safetensors_files


def scan_all_models_messages(base_models_path):
    """扫描所有模型并获取信息保存"""
    safetensors_files = scan_folders(base_models_path)
    messages_dict = {
        "models": {},
        "all_hash": [],
        "all_models_name": [],
        "all_models_path": []
    }
    model_message = {
        "hash": "",
        "path": "",
    }
    for safetensors_file in safetensors_files:
        model_base_name = os.path.basename(safetensors_file.replace('.safetensors', '').replace('.ckpt', ''))
        if model_base_name in messages_dict['all_models_name'] and safetensors_file not in messages_dict[
            'all_models_path']:
            warnings.warn(
                f"已存在同名模型模型：{messages_dict['models'][model_base_name]['path']},模型:{safetensors_file} 更新失败，请修改模型名称")
            continue

        model_message['hash'] = get_hash(safetensors_file)
        model_message['path'] = safetensors_file
        messages_dict['models'][model_base_name] = model_message.copy()
        print(f"已扫描模型：{model_base_name}")
        messages_dict['all_hash'].append(model_message['hash'])
        messages_dict['all_models_name'].append(model_base_name)
        messages_dict['all_models_path'].append(safetensors_file)

    with open(os.path.join(base_models_path, "models_messages.json"), "w", encoding="utf-8") as file:
        json.dump(messages_dict, file, ensure_ascii=False, indent=4)

    return messages_dict


def refresh_models_messages_by_path(base_models_path):
    if not os.path.exists(os.path.join(base_models_path, "models_messages.json")):
        print("未找到models_messages.json文件，正在创建...")
        scan_all_models_messages(base_models_path)

    """根据base路径在之前的日志基础上更新模型信息"""
    messages_dict = json.load(open(os.path.join(base_models_path, "models_messages.json"), "r", encoding="utf-8"))
    # 删除已经不存在的模型
    delmodels = []
    for model_name in messages_dict['all_models_name']:
        if not os.path.exists(messages_dict['models'][model_name]['path']):
            messages_dict['all_hash'].remove(messages_dict['models'][model_name]['hash'])
            messages_dict['all_models_name'].remove(model_name)
            messages_dict['all_models_path'].remove(messages_dict['models'][model_name]['path'])
            del messages_dict['models'][model_name]
            delmodels.append(model_name)
    # 添加新模型
    safetensors_files = scan_folders(base_models_path)
    newmodels = []
    for safetensors_file in safetensors_files:
        model_base_name = os.path.basename(safetensors_file.replace('.safetensors', '').replace('.ckpt', ''))
        if model_base_name not in messages_dict['all_models_name'] and safetensors_file not in messages_dict[
            'all_models_path']:
            model_message = {
                "hash": "",
                "path": "",
            }
            model_message['hash'] = get_hash(safetensors_file)
            model_message['path'] = safetensors_file
            messages_dict['models'][model_base_name] = model_message.copy()
            messages_dict['all_hash'].append(model_message['hash'])
            messages_dict['all_models_name'].append(model_base_name)
            messages_dict['all_models_path'].append(safetensors_file)
            newmodels.append(model_base_name)

        if model_base_name in messages_dict['all_models_name'] and safetensors_file not in messages_dict[
            'all_models_path']:
            warnings.warn(
                f"已存在同名模型模型：{messages_dict['models'][model_base_name]['path']},模型:{safetensors_file} 更新失败，请修改模型名称")

    if newmodels:
        print(f"新增模型：{newmodels}")
    else:
        print("没有新增模型")
    if delmodels:
        print(f"删除旧模型信息：{delmodels}")

    with open(os.path.join(base_models_path, "models_messages.json"), "w", encoding="utf-8") as file:
        json.dump(messages_dict, file, ensure_ascii=False, indent=4)

    return messages_dict


def download_models_by_hash(hash_value, model_path, model_name=None, api_key='986b5358d1e8897a9c3a66e7ea415672'):
    """根据哈希值下载模型"""
    try:
        api_url = f"https://civitai.com/api/v1/model-versions/by-hash/{hash_value}"
        response = requests.get(api_url)
        response.raise_for_status()
        if not model_name:
            model_name = response.json()['model']['name'] + '-' + response.json()['baseModel'] + '-' + response.json()[
                'name'] + '.safetensors'
        else:
            model_name = model_name + '.safetensors'
        d_path = os.path.join(model_path, model_name)
        print(f"正在下载模型: {model_name} 至 {d_path}")
        durl = response.json()['downloadUrl'] + f'?token={api_key}'
        d_response = requests.get(durl)
        d_response.raise_for_status()

        # 保存模型
        with open(d_path, 'wb') as f:
            f.write(d_response.content)

        print(f"已保存到: {d_path}")
        # 添加延时以避免API限制
        time.sleep(1)
        return True

    except Exception as e:
        print(f"处理hash {hash_value} 的模型时出错: {str(e)}")
        return False


def download_refreash_json_from_modelscope(base_models_path):
    os.system(f'modelscope download --dataset tangerboom/PretagsForBot pretags.json --local_dir {base_models_path}')


def refresh_models_messages_by_json(base_models_path, classed_dir=None,use_json_from_modelscope=True,local_json_path=None):
    """
    "classed_dir"例子：
    classed_dir={
    '人物':path1,
    '动作':path2,
    '服装':path3,
    '镜头':path4,
    '画风':path5,
    '场景':path6,
    '其他':path7,
    }
    """

    if not os.path.exists(os.path.join(base_models_path, "models_messages.json")):
        print("未找到models_messages.json文件，正在创建...")
        scan_all_models_messages(base_models_path)
    messages_dict = json.load(open(os.path.join(base_models_path, "models_messages.json"), "r", encoding="utf-8"))

    if use_json_from_modelscope:
        download_refreash_json_from_modelscope(base_models_path)
        pretags = json.load(open(os.path.join(base_models_path, "pretags.json"), "r", encoding="utf-8"))
    elif local_json_path:
        pretags = json.load(open(local_json_path, "r", encoding="utf-8"))
    else:
        raise Exception("请指定json文件")

    classes = list(pretags.keys())[:-1]
    if not classed_dir:
        classed_dir = {
            '人物': os.path.join(base_models_path, '人物'),
            '动作': os.path.join(base_models_path, '动作'),
            '服装': os.path.join(base_models_path, '服装'),
            '镜头': os.path.join(base_models_path, '镜头'),
            '画风': os.path.join(base_models_path, '画风'),
            '场景': os.path.join(base_models_path, '场景'),
            '其他': os.path.join(base_models_path, '其他'),
        }
    for class_name in classes:
        class_path = classed_dir[class_name]
        for modelname in pretags[class_name]:
            model = pretags[class_name][modelname]
            if model['model hash'] != 0 and model['model hash'] != "0":
                if model['model file name'] != 0 and model['model file name'] != "0" and model['model file name'] not in messages_dict[
                    'all_models_name']:
                    print(f'新模型：{model}')
                    model_message = {
                        "hash": "",
                        "path": "",
                    }
                    model_message['path'] = os.path.join(class_path, str(model['model file name']) + '.safetensors')
                    if not os.path.exists(class_path):
                        os.mkdir(class_path)

                    if not download_models_by_hash(model['model hash'], class_path, model['model file name']):
                        warnings.warn(f"下载模型失败：{model['model file name']}")
                        continue
                    else:
                        print(f"已保存到: {model_message['path']}")
    print("更新模型信息...")
    refresh_models_messages_by_path(base_models_path)


def run_LoraManager(base_models_path, command, classed_dir=None,use_json_from_modelscope=True,local_json_path=None):
    # 扫描base_models_path路径下的模型信息，计算模型哈希值，保存到models_messages.json中
    if command == '扫描全部':
        scan_all_models_messages(base_models_path)

    # 下载 pretags.json 并且根据json信息下载模型
    elif command == '下载更新':
        refresh_models_messages_by_json(base_models_path,classed_dir,use_json_from_modelscope,local_json_path)

    # 仅扫描base_models_path路径下的新模型，更新models_messages.json
    elif command == '本地更新':
        refresh_models_messages_by_path(base_models_path)


if __name__ == "__main__":
    # import os
    # os.environ['http_proxy'] = 'http://127.0.0.1:1090'
    # os.environ['https_proxy'] = 'http://127.0.0.1:1090'

    base_models_path = r'H:\StableAI_draw\stable-diffusion-webui\models\lora\LoraByTanger\char\琪亚娜-终焉原皮and泳装and新年'



    run_LoraManager(base_models_path, '下载更新', classed_dir=None,use_json_from_modelscope=True,local_json_path=None)
    # "classed_dir"是用来指定不同类型lora下载路径的，例子：
    # classed_dir={
    # '人物':path1,
    # '动作':path2,
    # '服装':path3,
    # '镜头':path4,
    # '画风':path5,
    # '场景':path6,
    # '其他':path7,
    # }
