# !/usr/bin/python3
# coding: utf-8

from crypt import methods
import os, json

import time
import subprocess
import zipfile

from flask import Flask, request, send_from_directory, make_response, render_template
from werkzeug.utils import secure_filename

from exp_runner import Exp_Runner

app = Flask(__name__)

## 设置上传数据保存文件夹,即多视图三维重建项目加载数据路径
app.config['UPLOAD_FOLDER'] = './data/'

## 设置三维重建项目输出结果存放路径
app.config['RESULTS_FOLDER'] = './final_out/'

## 设置待下载文件压缩包存放路径
app.config['DOWNLOAD_ZIP_FOLDER'] = './download_zips'
os.makedirs(app.config['DOWNLOAD_ZIP_FOLDER'], exist_ok=True)

## 设置允许上传的文件格式
ALLOW_EXTENSIONS = ['zip',]

## 判断文件后缀是否在列表中
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[-1] in ALLOW_EXTENSIONS

## 检查static/目录, 不存在则创建, 存在则清空
def check_folder(folder):
    if os.path.exists(folder) and len(os.listdir(folder)) > 0:
        FNULL = open(os.devnull, 'w')
        subprocess.call(f"rm {folder}/*.*", shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    else:
        os.makedirs(folder, exist_ok=True)

## 将上传的图片移动到指定目录下
def run_copy(src_img, dst_img):
    strcmd = 'cp ' + src_img + ' ' + dst_img
    subprocess.call(strcmd, shell=True)

## 打包zip文件
def make_zip(filepath, source_dir):
    zipf = zipfile.ZipFile(source_dir, 'w')
    pre_len = len(os.path.dirname(filepath))
    for parent, dirnames, filenames in os.walk(filepath):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)
            zipf.write(pathfile, arcname)
    
    zipf.close()

## 解压zip文件
def ext_zip(filepath, dst_dir):
    zipf = zipfile.ZipFile(filepath, 'r')
    for file in zipf.namelist():
        zipf.extract(file, dst_dir)
    
    zipf.close()
    ## 删除zip文件
    # strcmd = 'rm ' + filepath
    # subprocess.call(strcmd, shell=True)

## 查找指定目录中最新的文件夹
def find_newest_folder(m_folder):
    return max([os.path.join(m_folder, d) for d in os.listdir(m_folder)], key=os.path.getmtime)

@app.route('/')
def index():
    return make_response(render_template('index.html'))


## 自动化三维重建
@app.route('/api/auto_reconstruct', methods=['POST', 'GET'])
def auto_reconstruct():
    if request.method == 'POST':
        file_data = request.files.get('file')
        state     = request.form.get('state')
        file_name = file_data.filename

        ### ste p01: 设置数据集加载路径并加载数据
        date_name   = time.strftime('%Y-%m-%d')
        data_folder = os.path.join(app.config['UPLOAD_FOLDER'], date_name)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder, exist_ok=True)

        if file_data and allowed_file(file_name):
            file_name = secure_filename(file_name)
            ext_zip(file_data, data_folder)                                                # 将上传的压缩包文件解压到数据集加载目录下

        case_name  = find_newest_folder(data_folder).split('/')[-1]
        out_folder = os.path.join(app.config['RESULTS_FOLDER'], date_name, case_name)

        exp_runner = Exp_Runner(data_folder, case_name, out_folder, checkpoint=None)
        
        ### step 02：进行自动化三维重建，可通过指定state参数选择单步重建
        print('=' * 90)
        if len(state) > 0:
            print('=> Start One-step reconstruction...')
            print('-' * 80)
            ## Step 02-1: 通过colmap估计位姿参数文件poses.npy
            if state == 'colmap':
                print('=> Step-01: start colmap poses estimation...')
                print('-' * 70)
                exp_runner.gen_poses()
                print('-' * 90)

                return {"code": '200', "data": "", "message": "colmap estimation completed!"}

            ## Step 02-2: 训练模式(默认mode=='train')训练训练网络，生成mesh模型
            elif state == 'train':
                print('=> Step-02: start training mesh...')
                print('-' * 70)
                exp_runner.mode = 'train'
                exp_runner.mesh_train()
                print('-' * 90)

                return {"code": '200', "data": "", "message": "mesh training completed!"}

            ## Step 02-3: 验证模式, mode=='validate_mesh'， 生成并提取mesh模型
            elif state == 'validate':
                print('=> Step-03: start validating and get mesh file...')
                print('-' * 70)
                exp_runner.mode = 'validate_mesh'
                exp_runner.mesh_extract()
                print('-' * 90)
                
                return {"code": '200', "data": "", "message": "mesh extract successful!"}

            ## Step 02-4: 基于02-3得到的mesh白膜，训练并生成对应的贴图(kd, ks, n)以及光照probe.hdr等信息
            elif state == 'texture':
                print('=> Step-04: start training texture...')
                print('-' * 70)
                exp_runner.gen_texture()
                print('-' * 90)
                
                return {"code": '200', "data": "", "message": "texture generate successful!"}
        else:
            ## 进行自动化三维重建
            print('=> Start Auto reconstruction...')
            print('-' * 80)
            exp_runner.auto_reconstruct()
            print('=> All Done! ------------------------')
            print('=' * 90)
        
        ### step 03：重建结果输出与下载
        zip_name = case_name + '.zip'

        ## 将待下载的所有文件打包为zip文件
        down_zip_folder = os.path.join(app.config['DOWNLOAD_ZIP_FOLDER'], date_name)
        if not os.path.exists(down_zip_folder):
            os.makedirs(down_zip_folder, exist_ok=True)
        make_zip(out_folder, os.path.join(down_zip_folder, zip_name))
        
        return {"code": '200', "data": os.path.join(down_zip_folder, zip_name), "message": "Auto reconstruction successful!"}

    else:
        return {"code": '503', "data": "", "message": "only support post method!"}

## 结果文件下载
@app.route('/api/auto_reconstruct/download_file', methods=['GET', 'POST'])
def download_file():
    if request.method == 'POST':

        final_zipname = request.form.get('f_name')
        final_zipdate = request.form.get('f_date')

        ## 根据指定的文件名与日期，判断该文件是否存在，存在则下载，不存在则返回提示信息
        download_zip_name   = final_zipname + '.zip'
        downlowd_zip_folder = os.path.join(app.config['DOWNLOAD_ZIP_FOLDER'], final_zipdate)

        if not os.path.exists(os.path.join(downlowd_zip_folder, download_zip_name)):
            return {"code": '404', "data": "NOT FOUND", "message": "The requested resource does not exist!"}         # 判断待下载文件是否存在
        else:
            return make_response(send_from_directory(downlowd_zip_folder, download_zip_name, as_attachment=True))    # 存在则下载文件
    
    else:
        return {"code": '503', "data": "", "message": "only support post method!"}


## 上传数据
@app.route('/api/auto_reconstruct/upload_data', methods=['POST', 'GET'])
def upload_data():
    if request.method == 'POST':
        filedata = request.files.get('file')                                             # 获取post过来的文件
        filename = filedata.filename                                                     # 获取文件名称

        files_dir = time.strftime('%Y-%m-%d')
        saved_dir = os.path.join(app.config['UPLOAD_FOLDER'], files_dir)
        if not os.path.exists(saved_dir):
            os.makedirs(saved_dir, exist_ok=True)
        
        if filedata and allowed_file(filename):
            filename = secure_filename(filename)
            ext_zip(filedata, saved_dir)       
        
            return {"code": '200', "data": "", "message": "{} upload successful!".format(filename)}
        
        else:
            return ("format error, only support .zip file format!")

    else:
        return {"code": '503', "data": "", "message": "only support post method!"}


## 多视图三维重建及贴图生成
@app.route('/api/auto_reconstruct/reconstruct', methods=['POST', 'GET'])
def NeuS_Texture():
    if request.method == 'POST':
        # case_name = request.json.get('case_name')
        state = request.form.get('state')
        base_folder = os.path.join(app.config['UPLOAD_FOLDER'], time.strftime('%Y-%m-%d'))
        case_name = find_newest_folder(base_folder).split('/')[-1]
        
        out_dir = os.path.join(app.config['RESULTS_FOLDER'], case_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        exp_runner = Exp_Runner(case_name, out_dir, checkpoint=None)
        print('=' * 90)
        ### Step 01: 通过colmap估计位姿参数文件poses.npy
        if state == 'colmap':
            print('=> Step-01: start colmap poses estimation...')
            print('-' * 70)
            exp_runner.gen_poses()
            print('-' * 90)

            return {"code": '200', "data": "", "message": "colmap estimation completed!"}

        ### Step 02: 设置训练模式(默认mode=='train')训练训练网络，生成mesh模型
        elif state == 'train':
            print('=> Step-02: start training mesh...')
            print('-' * 70)
            exp_runner.mode = 'train'
            exp_runner.mesh_train()
            print('-' * 90)

            return {"code": '200', "data": "", "message": "mesh training completed!"}

        ### Step 03: 设置验证模式, mode=='validate_mesh'， 生成并提取mesh模型
        elif state == 'validate':
            print('=> Step-03: start validating and get mesh file...')
            print('-' * 70)
            exp_runner.mode = 'validate_mesh'
            exp_runner.mesh_extract()
            print('-' * 90)
            
            return {"code": '200', "data": "", "message": "mesh extract successful!"}

        ### Step 04: 基于step 03得到的mesh白膜，训练并生成对应的贴图(kd, ks, n)以及光照probe.hdr等信息
        elif state == 'texture':
            print('=> Step-04: start training texture...')
            print('-' * 70)
            exp_runner.gen_texture()
            print('-' * 90)
            
            return {"code": '200', "data": "", "message": "texture generate successful!"}

        
        ### 自动化三维重建与贴图生成
        else:
            print('=> Start Auto reconstruct...')
            exp_runner.auto_reconstruct()
            print('=> All Done! ------------------------')
            print('=' * 90)

            return {"code": '200', "data": out_dir, "message": "reconstruct successful!"}

    else:
        return {"code": '503', "data": "", "message": "only support post method!"}


if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=10423, debug=True)
    #app.run(debug=True)