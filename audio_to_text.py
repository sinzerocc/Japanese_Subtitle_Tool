import os
import sys
import webbrowser
import json
from http.server import SimpleHTTPRequestHandler, HTTPServer
import threading
import torch
import whisper
import time

# 全局状态控制中心
status_log = ["⚙️ 系统初始化成功，4060 显卡就绪，等待任务中..."]
is_running = False
current_progress = 0  # 实时百分比 0 - 100
start_time_stamp = 0
estimated_duration = 0 

def append_log(msg):
    print(msg)
    status_log.append(msg)

def track_progress_loop():
    global current_progress, is_running, start_time_stamp, estimated_duration
    stage1_lines = 120
    for _ in range(stage1_lines):
        if not is_running: return
        if current_progress < 15:
            current_progress += 0.15
            time.sleep(0.1)
            
    while is_running:
        if current_progress >= 95:
            time.sleep(0.5)
            continue
        elapsed = time.time() - start_time_stamp
        if estimated_duration > 0:
            calc_prog = 15 + int((elapsed / estimated_duration) * 80)
            if calc_prog > current_progress:
                current_progress = min(calc_prog, 95)
        time.sleep(0.5)

def run_whisper_core(audio_file_path, fmt):
    global is_running, current_progress, start_time_stamp, estimated_duration
    is_running = True
    current_progress = 0
    start_time_stamp = time.time()
    
    output_file = os.path.splitext(audio_file_path)[0] + f"_result.{fmt}"
    
    try:
        append_log("🔄 正在加载 Whisper medium 模型并初始化显存...")
        model = whisper.load_model("medium", device="cuda")
        estimated_duration = 45  
        
        threading.Thread(target=track_progress_loop, daemon=True).start()
        
        append_log(f"🚀 4060 显卡全开！开始解析日语语义...")
        result = model.transcribe(audio_file_path, language="ja")
        
        append_log("✍️ 正在将识别结果写出到本地文件...")
        with open(output_file, "w", encoding="utf-8") as f:
            if fmt == "txt":
                for segment in result['segments']:
                    start = int(segment['start'])
                    timestamp = f"[{start // 60:02d}:{start % 60:02d}] "
                    f.write(f"{timestamp}{segment['text']}\n")
            elif fmt == "srt":
                for i, segment in enumerate(result['segments'], start=1):
                    def fmt_time(seconds):
                        hours = int(seconds // 3600)
                        minutes = int((seconds % 3600) // 60)
                        secs = int(seconds % 60)
                        ms = int((seconds % 1) * 1000)
                        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
                    start_str = fmt_time(segment['start'])
                    end_str = fmt_time(segment['end'])
                    f.write(f"{i}\n{start_str} --> {end_str}\n{segment['text']}\n\n")
                    
        current_progress = 100
        append_log(f"🎉 成功！文本已成功保存在：\n{output_file}")
    except Exception as e:
        append_log(f"❌ 发生错误: {str(e)}")
    finally:
        is_running = False

def convert_txt_to_srt_core(txt_file_path):
    """核心离线转换逻辑：解析带时间戳的 txt，直接吐出 srt，耗时 0.1 秒"""
    if not os.path.exists(txt_file_path):
        return False, "找不到指定的 TXT 文件！"
    
    try:
        srt_file_path = txt_file_path.replace("_result.txt", "").replace(".txt", "") + "_converted.srt"
        with open(txt_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        parsed_segments = []
        for line in lines:
            line = line.strip()
            if not line.startswith("[") or "]" not in line:
                continue
            
            # 拆分时间戳和文本 [01:23] 文本内容
            time_part, text_part = line.split("]", 1)
            time_str = time_part.replace("[", "").strip() # "01:23"
            text_part = text_part.strip()
            
            # 解析成秒数
            parts = time_str.split(":")
            if len(parts) == 2:
                seconds = int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            else:
                continue
                
            parsed_segments.append({"start": seconds, "text": text_part})
        
        if not parsed_segments:
            return False, "TXT 文件格式不符，未发现可解析的 [分:秒] 时间戳格式！"
            
        # 写入 SRT
        with open(srt_file_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(parsed_segments, start=1):
                start_sec = seg["start"]
                # 智能预估结束时间：如果没有下一句，默认撑 3 秒；如果有，则持续到下一句开始
                end_sec = parsed_segments[i]["start"] if i < len(parsed_segments) else start_sec + 3
                
                def fmt_time(seconds):
                    hours = int(seconds // 3600)
                    minutes = int((seconds % 3600) // 60)
                    secs = int(seconds % 60)
                    return f"{hours:02d}:{minutes:02d}:{secs:02d},000"
                
                f.write(f"{i}\n{fmt_time(start_sec)} --> {fmt_time(end_sec)}\n{seg['text']}\n\n")
                
        return True, srt_file_path
    except Exception as e:
        return False, str(e)

class WebHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            
            device_text = "🚀 已启用 NVIDIA 4060 显卡加速 (CUDA)"
            device_color = "#28a745"
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>日语音频转文字工具 (4060 进度条强化版)</title>
                <style>
                    body {{ font-family: -apple-system, sans-serif; background: #f5f5f7; padding: 20px; color: #333; }}
                    .card {{ background: white; max-width: 650px; margin: 0 auto 20px auto; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
                    h2 {{ margin-top: 0; color: #1d1d1f; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                    .device {{ font-weight: bold; color: {device_color}; margin-bottom: 20px; font-size: 14px; }}
                    .form-group {{ margin-bottom: 18px; }}
                    label {{ display: block; font-weight: bold; margin-bottom: 6px; font-size: 14px; }}
                    .file-input-wrapper {{ display: flex; flex-direction: column; gap: 8px; }}
                    input[type="file"] {{ padding: 12px; border: 1px dashed #ccc; border-radius: 6px; background: #fafafa; font-size: 14px; cursor: pointer; }}
                    input[type="text"] {{ padding: 10px; box-sizing: border-box; border: 1px solid #ccc; border-radius: 6px; font-size: 14px; background: #eee; color: #555; }}
                    select {{ padding: 10px; width: 100%; border: 1px solid #ccc; border-radius: 6px; font-size: 14px; background: white; }}
                    .btn-run {{ width: 100%; padding: 12px; background: #fc3c44; color: white; border: none; border-radius: 6px; font-size: 16px; font-weight: bold; cursor: pointer; margin-top: 10px; }}
                    .btn-tools {{ width: 100%; padding: 12px; background: #34a853; color: white; border: none; border-radius: 6px; font-size: 15px; font-weight: bold; cursor: pointer; margin-top: 10px; }}
                    button:disabled {{ background: #ccc; cursor: not-allowed; }}
                    
                    .progress-container {{ margin-top: 20px; background: #e5e5ea; border-radius: 8px; overflow: hidden; display: none; box-shadow: inset 0 1px 2px rgba(0,0,0,0.1); }}
                    .progress-bar {{ width: 0%; height: 16px; background: linear-gradient(90deg, #0071e3, #34a853); transition: width 0.4s ease; }}
                    .progress-text {{ font-size: 14px; font-weight: bold; color: #0071e3; margin-top: 8px; text-align: center; display: none; }}
                    
                    #log {{ background: #222; color: #00ff00; padding: 15px; height: 160px; overflow-y: auto; border-radius: 6px; font-family: monospace; font-size: 12px; margin-top: 15px; white-space: pre-line; }}
                </style>
            </head>
            <body>
                <div class="card">
                    <h2>🎙️ 日语音频转文字工具 (4060 专属 Web 版)</h2>
                    <div class="device">{device_text}</div>
                    
                    <div class="form-group">
                        <label>1. 选择播客音频:</label>
                        <div class="file-input-wrapper">
                            <input type="file" id="filePicker" accept=".mp3,.m4a,.wav,.aac,.flac" onchange="handleFileSelect()">
                            <input type="text" id="audioPath" readonly placeholder="选择文件后此处会自动解析文件名...">
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>2. 选择导出格式:</label>
                        <select id="fmt">
                            <option value="txt">纯文本带简易时间戳 (.txt)</option>
                            <option value="srt">标准播放器外挂字幕 (.srt)</option>
                        </select>
                    </div>
                    
                    <button id="startBtn" class="btn-run" onclick="startTask()">🔥 一键开始转换</button>
                    
                    <div class="progress-container" id="pContainer"><div class="progress-bar" id="pBar"></div></div>
                    <div class="progress-text" id="pText">准备就绪: 0%</div>
                    
                    <label style="margin-top:20px; display:block; font-weight:bold;">运行状态与日志:</label>
                    <div id="log"></div>
                </div>

                <div class="card" style="border: 1px solid #34a853; background: #fafdfb;">
                    <h2 style="color: #34a853;">🛠️ 附加功能：已有 TXT 秒转 SRT 字幕</h2>
                    <div class="form-group">
                        <label>选择之前转好的 _result.txt 文件:</label>
                        <div class="file-input-wrapper">
                            <input type="file" id="txtPicker" accept=".txt" onchange="handleTxtSelect()">
                            <input type="text" id="txtPath" readonly placeholder="请选择需要提取字幕的 txt 文本...">
                        </div>
                    </div>
                    <button id="convertBtn" class="btn-tools" onclick="startTxtConvert()">⚡ 智能提取，瞬间秒转 SRT</button>
                </div>

                <script>
                    let selectedFileObject = null;
                    let selectedTxtObject = null;

                    function updateLog() {{
                        fetch('/get_log')
                            .then(res => res.json())
                            .then(data => {{
                                const logDiv = document.getElementById('log');
                                logDiv.innerText = data.logs.join('\\n');
                                logDiv.scrollTop = logDiv.scrollHeight;
                                
                                document.getElementById('startBtn').disabled = data.running;
                                
                                if(data.running) {{
                                    document.getElementById('startBtn').innerText = "⚡ 正在努力啃音频中...";
                                    document.getElementById('pContainer').style.display = 'block';
                                    document.getElementById('pText').style.display = 'block';
                                    
                                    let displayProg = Math.floor(data.progress);
                                    document.getElementById('pBar').style.width = displayProg + '%';
                                    document.getElementById('pText').innerText = "⚡ 4060 加速转换中: " + displayProg + "%";
                                }} else {{
                                    document.getElementById('startBtn').innerText = "🔥 一键开始转换";
                                    if(data.progress >= 100) {{
                                        document.getElementById('pBar').style.width = '100%';
                                        document.getElementById('pBar').style.background = '#28a745';
                                        document.getElementById('pText').innerHTML = "🎉 <span style='color:#28a745'>转换成功！100% 文本已输出完成</span>";
                                    }}
                                }}
                            }});
                    }}
                    setInterval(updateLog, 1000);

                    function handleFileSelect() {{
                        const fileInput = document.getElementById('filePicker');
                        if (fileInput.files.length > 0) {{
                            selectedFileObject = fileInput.files[0];
                            document.getElementById('audioPath').value = selectedFileObject.name;
                        }}
                    }}

                    function handleTxtSelect() {{
                        const txtInput = document.getElementById('txtPicker');
                        if (txtInput.files.length > 0) {{
                            selectedTxtObject = txtInput.files[0];
                            document.getElementById('txtPath').value = selectedTxtObject.name;
                        }}
                    }}

                    function startTask() {{
                        const fmt = document.getElementById('fmt').value;
                        if(!selectedFileObject) {{ alert('请先选择一个音频文件！'); return; }}
                        
                        const formData = new FormData();
                        formData.append('audio_file', selectedFileObject);
                        formData.append('fmt', fmt);
                        
                        document.getElementById('startBtn').disabled = true;
                        
                        fetch('/upload_and_start', {{
                            method: 'POST',
                            body: formData
                        }}).then(res => res.json()).then(data => {{
                            if(!data.success) alert(data.msg);
                        }});
                    }}

                    function startTxtConvert() {{
                        if(!selectedTxtObject) {{ alert('请先选择一个文本文件！'); return; }}
                        
                        const formData = new FormData();
                        formData.append('txt_file', selectedTxtObject);
                        
                        fetch('/convert_txt', {{
                            method: 'POST',
                            body: formData
                        }}).then(res => res.json()).then(data => {{
                            if(data.success) {{
                                alert("🎉 秒转成功！\\n字幕已保存在本地同目录下，文件名为:\\n" + data.filename);
                            }} else {{
                                alert("❌ 转换失败: " + data.msg);
                            }}
                        }});
                    }}
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
            
        elif self.path == '/get_log':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {"logs": status_log, "running": is_running, "progress": current_progress}
            self.wfile.write(json.dumps(response).encode('utf-8'))
            
    def do_POST(self):
        if self.path == '/upload_and_start':
            try:
                content_length = int(self.headers['Content-Length'])
                boundary = self.headers['Content-Type'].split("=")[1].encode()
                body = self.rfile.read(content_length)
                
                fn_start = body.find(b'filename="') + 10
                fn_end = body.find(b'"', fn_start)
                filename = body[fn_start:fn_end].decode('utf-8')
                
                head_end = body.find(b'\r\n\r\n', fn_end) + 4
                file_data = body[head_end:body.find(b'\r\n--' + boundary, head_end)]
                
                fmt = "txt"
                if b'name="fmt"' in body:
                    fmt_start = body.find(b'\r\n\r\n', body.find(b'name="fmt"')) + 4
                    fmt = body[fmt_start:body.find(b'\r\n', fmt_start)].decode('utf-8').strip()

                target_path = os.path.join(os.getcwd(), filename)
                with open(target_path, 'wb') as save_f:
                    save_f.write(file_data)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                if is_running:
                    self.wfile.write(json.dumps({"success": False, "msg": "当前已有任务在运行！"}).encode('utf-8'))
                    return
                
                global status_log
                status_log.clear()
                append_log(f"🚀 成功接收音频文件并载入流: {filename}")
                
                threading.Thread(target=run_whisper_core, args=(target_path, fmt), daemon=True).start()
                self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
                
            except Exception as e:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "msg": f"文件解析失败: {str(e)}"}).encode('utf-8'))

        elif self.path == '/convert_txt':
            # 接收前端上传上来的纯文本文件，直接做秒级格式化转换
            try:
                content_length = int(self.headers['Content-Length'])
                boundary = self.headers['Content-Type'].split("=")[1].encode()
                body = self.rfile.read(content_length)
                
                fn_start = body.find(b'filename="') + 10
                fn_end = body.find(b'"', fn_start)
                filename = body[fn_start:fn_end].decode('utf-8')
                
                head_end = body.find(b'\r\n\r\n', fn_end) + 4
                file_data = body[head_end:body.find(b'\r\n--' + boundary, head_end)]
                
                # 存入本地临时解析
                temp_txt_path = os.path.join(os.getcwd(), filename)
                with open(temp_txt_path, 'wb') as tmp_f:
                    tmp_f.write(file_data)
                
                # 调用核心本地秒转函数
                success, result_msg = convert_txt_to_srt_core(temp_txt_path)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                
                if success:
                    # 返回生成的 SRT 文件名
                    self.wfile.write(json.dumps({"success": True, "filename": os.path.basename(result_msg)}).encode('utf-8'))
                else:
                    self.wfile.write(json.dumps({"success": False, "msg": result_msg}).encode('utf-8'))
            except Exception as e:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "msg": str(e)}).encode('utf-8'))

def start_server():
    server = HTTPServer(('127.0.0.1', 8989), WebHandler)
    print("🌍 带有离线秒转功能的控制台已拉起...")
    webbrowser.open("http://127.0.0.1:8989")
    server.serve_forever()

if __name__ == "__main__":
    start_server()