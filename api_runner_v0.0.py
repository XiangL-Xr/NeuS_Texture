# !/usr/bin/python3
# coding: utf-8

import os, json
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import subprocess
import zipfile

from flask import Flask, request, send_from_directory, make_response
from werkzeug.utils import secure_filename

from exp_runner import Exp_Runner

app = Flask(__name__)

## 设置上传数据保存文件夹,即多视图三维重建项目加载数据路径
app.config['UPLOAD_FOLDER'] = './data/'

## 设置三维重建项目输出结果存放路径
app.config['RESULTS_FOLDER'] = './final_out/'

## 设置待下载文件压缩包存放路径
app.config['DOWNLOAD_ZIP_FOLDER'] = '.download_zips'
os.makedirs(app.config['DOWNLOAD_ZIP_FOLDER'], exist_ok=True)

## 设置允许上传的文件格式
ALLOW_EXTENSIONS = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png']

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

## 上传图片
@app.route('/api/neus_texture/upload_data', methods=['POST', 'GET'])
def upload_data():
    if request.method == 'POST':
        filedata = request.files.get('photo')                                             # 获取post过来的图片数据
        casename = request.form.get('case_name')                                          # 获取自定义商品名称
        filename = filedata.filename                                                      # 获取图片名称
        filetype = filedata.content_type                                                  # 获取图片类型，image or mask
        
        ## 创建重建项目加载数据目录
        if filetype == 'image':
            data_folder = os.path.join(app.config['UPLOAD_FOLDER'], casename, 'images')  
        elif filetype == 'mask':
            data_folder = os.path.join(app.config['UPLOAD_FOLDER'], casename, 'masks')
        
        if not os.path.exists(data_folder):
            os.makedirs(data_folder, exist_ok=True)
        
        # check_folder(images_folder)                                                    # 检查图片上传目录
        ## 检测文件格式
        if filedata and allowed_file(filename):
            # file_name = secure_filename(filename)                                      # secure_filename方法用来去掉文件名中的中文
            filedata.save(os.path.join(data_folder, filename))                           # 保存图片到指定文件夹
            
            return {"code": '200', "data": filename, "message": "{} upload successful!".format(filetype)}
        
        else:
            return ("format error, only support jpg, jpeg, png file format!")

    else:
        return {"code": '503', "data": "", "message": "only support post method!"}


## 多视图三维重建及贴图生成
@app.route('/api/neus_texture/reconstruct', methods=['POST', 'GET'])
def NeuS_Texture():
    if request.method == 'POST':
        case_name = request.json.get('case_name')
        state = request.json.get('state')
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
        elif state == 'auto':
            exp_runner.auto_reconstruct()
            return {"code": '200', "data": out_dir, "message": "reconstruct successful!"}
        
        else:
            print('=> State Error, Please select the correct run state!')

        print('=> All Done! ------------------------')
        print('=' * 90)
       

    else:
        return {"code": '503', "data": "", "message": "only support post method!"}


@app.route('/api/neus_texture/download_files', methods=['GET', 'POST'])
def download_file():
    if request.method == 'POST':
        print('=' * 70)
        case_name = request.json.get('case_name')
        down_type = request.json.get('download_type')

        ## 根据指定的下载类型，选择待下载文件路径
        if down_type == 'all':
            files_dir = os.path.join(app.config['RESULTS_FOLDER'], case_name)
            zip_name = case_name + '.zip'
        elif down_type == 'mesh':
            files_dir = os.path.join(app.config['RESULTS_FOLDER'], case_name, 'mesh')
            zip_name = case_name + '_mesh.zip'
        elif down_type == 'render_results':
            files_dir = os.path.join(app.config['RESULTS_FOLDER'], case_name, 'validate')
            zip_name = case_name + '_render_results.zip'
        else:
            print('=> Download type Error, Please select the correct downlaod type!')

        ## 将待下载的所有文件打包为zip文件
        make_zip(files_dir, os.path.join(app.config['DOWNLOAD_ZIP_FOLDER'], zip_name))
        
        ## 下载文件
        response = make_response(send_from_directory(app.config['DOWNLOAD_ZIP_FOLDER'], zip_name, as_attachment=True))
        response.headers["filename"] = "{}".format(zip_name)
        response.headers["m_text"]   = {
                            "code": '200', 
                            "data": "zip_name", 
                            "message": "zipfile download successful!"
                        }
        # return send_file(memory_file, download_name=zip_name, mimetype='zip', as_attachment=True)
        
        return response
    
    else:
        return {"code": '503', "data": "", "message": "only support post method!"}


if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=10423, debug=True)
    # app.run(debug=True)