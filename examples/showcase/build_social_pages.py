"""Build two shareable PDF pages from published results (requires reportlab).

Use --font / --bold-font for licensed Chinese TrueType fonts; fonts are not
distributed. Render PNGs using pdftoppm -r 144 -png learning-pages.pdf page.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from reportlab.graphics import renderPDF
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.shapes import Drawing
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[2]
W, H = 720, 1080
INK, MUTED, BG = '#141719', '#5A6266', '#FAF7F0'
BLUE, TEAL, YELLOW = '#164DDD', '#008779', '#FFE642'


class Page:
    def __init__(self, c):
        self.c = c

    def rect(self, x, y, w, h, fill, radius=14):
        self.c.setFillColor(HexColor(fill))
        self.c.roundRect(x, H-y-h, w, h, radius, stroke=0, fill=1)

    def text(self, text, x, y, size=16, color=INK, font='CN', max_width=None):
        if max_width and pdfmetrics.stringWidth(text, font, size) > max_width:
            raise ValueError(f'Text exceeds layout width: {text}')
        self.c.setFillColor(HexColor(color))
        self.c.setFont(font, size)
        self.c.drawString(x, H-y-size, text)

    def lines(self, lines, x, y, size=16, color=INK, leading=26, font='CN', max_width=600):
        for i, line in enumerate(lines):
            self.text(line, x, y+i*leading, size, color, font, max_width)

    def base(self, number, label):
        self.rect(0, 0, W, H, BG, 0)
        self.rect(36, 34, 244, 33, YELLOW, 16)
        self.text(f'开源 AI 项目 / {label}', 50, 40, 16, font='CNB')
        self.text(number, 638, 36, 23, BLUE, 'Helvetica-Bold')
        self.c.setStrokeColor(HexColor('#D9DAD4'))
        self.c.line(36, 55, 684, 55)
        self.text('MULTIMODAL RETRIEVAL LAB', 36, 1034, 10, MUTED, 'Helvetica-Bold')
        self.text('项目记录 / 非通用能力排名', 485, 1030, 12, MUTED)


def data():
    csv_path = ROOT/'docs/study/comparison.csv'
    json_path = ROOT/'docs/study/controls/qwen-controls.json'
    with csv_path.open() as stream:
        fixed = {(r['variant'], r['mode']): float(r['primary_five_calibrated_f1'])
                 for r in csv.DictReader(stream) if r['scope'] == 'full'}
    controls = json.loads(json_path.read_text())
    norm = {r['variant']: {m: r['metrics'][m]['f1'] for m in ('text', 'image')}
            for r in controls['results'] if r['control'] == 'normalized'}
    metrics = {
        'qwen': {m: [norm['qwen-base'][m]*100, norm['qwen-best'][m]*100] for m in ('text', 'image')},
        'embedding': {m: [fixed['embedding-base', m]*100, fixed['embedding-best', m]*100] for m in ('text', 'image')},
    }
    return metrics, {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (csv_path, json_path)}


def metric_card(p, y, title, subtitle, values, color):
    p.rect(36, y, 648, 197, '#FFFFFF')
    p.rect(36, y, 7, 197, color, 3)
    p.text(title, 57, y+14, 23, color, 'CNB')
    p.text(subtitle, 57, y+47, 12, MUTED, max_width=600)
    p.text('检索 F1', 58, y+74, 12, MUTED)
    p.text('基座', 200, y+74, 12, MUTED)
    p.text('微调', 375, y+74, 12, MUTED)
    p.text('提升 / 百分点', 532, y+74, 12, MUTED)
    for i, (mode, label) in enumerate((('text', '文搜图'), ('image', '图搜图'))):
        base, tuned = values[mode]
        row = y+100+i*44
        p.text(label, 58, row+5, 18, font='CNB')
        p.text(f'{base:.2f}%', 184, row, 26, MUTED, 'Helvetica')
        p.text('→', 316, row+3, 21, MUTED)
        p.text(f'{tuned:.2f}%', 361, row, 28, color, 'Helvetica-Bold')
        p.rect(525, row-1, 136, 37, '#F1F4EF', 7)
        p.text(f'+{tuned-base:.2f}', 543, row+3, 24, color, 'Helvetica-Bold')


def result_page(c, metrics):
    p = Page(c)
    p.base('03', '训练与验证')
    p.text('微调到底有没有用？', 36, 86, 39, font='CNB', max_width=650)
    p.text('让同一图库说话', 36, 139, 42, font='CNB')
    p.text('先对齐评测口径，再讨论提升来自哪里。', 38, 198, 18, MUTED)
    p.rect(36, 240, 648, 66, INK)
    p.text('4,500 / 1,000 / 2,000', 54, 250, 24, YELLOW, 'Helvetica-Bold')
    p.text('训练 / 验证 / 测试图片', 54, 280, 12, '#FFFFFF')
    p.text('8 组配置 · 24 个验证检查点', 392, 254, 17, '#FFFFFF', 'CNB', 274)
    p.text('模型与阈值只在验证集选', 392, 281, 12, '#DDE2E0')
    metric_card(p, 326, 'A / Qwen LoRA', '统一 JSON 结构归一化后的补充对照；非新盲测', metrics['qwen'], BLUE)
    metric_card(p, 541, 'B / Embedding 对比学习', '原始固定协议；直接编码原图，不经过 Qwen 类别筛选', metrics['embedding'], TEAL)
    p.rect(36, 757, 648, 157, '#EEEFE8')
    p.text('涨点之外，我还追查了原因', 54, 772, 23, font='CNB')
    p.lines([
        'Qwen 的差距包含格式适配，不能全部算成视觉能力。',
        '共用类别筛选、只换描述向量：+2.48 / +0.01 个百分点。',
        '在这组类别检索实验中，主要收益集中在类别筛选。',
    ], 54, 810, 16, leading=29, max_width=608)
    p.lines([
        '范围：15 类候选图库，报告五个重点类别的平均 F1。',
        '两组协议分别解读；单随机种子，不代表细粒度或同款匹配。',
        '来源：docs/study/comparison.csv · docs/QWEN_CONTROLS.md',
    ], 38, 936, 13, MUTED, 23, max_width=647)
    c.showPage()


def step(p, y, number, title, subtitle):
    p.rect(36, y, 648, 166, '#FFFFFF')
    p.rect(53, y+17, 37, 35, YELLOW, 8)
    p.text(number, 60, y+21, 21, INK, 'Helvetica-Bold')
    p.text(title, 102, y+17, 23, font='CNB', max_width=557)
    p.text(subtitle, 103, y+51, 13, MUTED, max_width=550)


def start_page(c):
    p = Page(c)
    p.base('04', '运行与学习')
    p.text('先跑通，再换数据', 36, 87, 43, font='CNB', max_width=650)
    p.text('最后训练自己的模型', 36, 143, 38, font='CNB', max_width=650)
    p.text('从看得见的结果，走到可解释的实验。', 38, 199, 18, MUTED)
    step(p, 241, '1', '打开演示 / 无需 GPU', '下载仓库，在项目根目录执行；只需 Python 3.10+')
    p.rect(54, 318, 610, 43, INK, 7)
    p.text('python3 examples/quickstart.py', 70, 328, 20, '#FFFFFF', 'Courier', 577)
    p.text('这是已记录结果的回放；不是实时模型推理。', 55, 376, 14, MUTED)
    step(p, 425, '2', '换自己的图片 / 真正跑检索', '有模型运行环境后，先用仓库里的 12 张示例图体验')
    p.text('准备图片与清单  →  建索引  →  输入查询', 56, 506, 21, TEAL, 'CNB', 610)
    p.text('公开配置使用基座；微调权重尚未分发。', 56, 550, 14, MUTED)
    step(p, 609, '3', '做训练 / 比较两条技术路线', '先划分训练、验证、测试，再构建监督样本')
    p.text('Qwen LoRA   ×   Embedding InfoNCE', 56, 690, 23, BLUE, 'CNB', 610)
    p.text('验证集选参数，固定测试集报告 F1 / mAP。', 56, 734, 14, MUTED)
    p.rect(36, 794, 648, 206, INK)
    p.text('手机扫码，或点击链接直接打开', 55, 814, 23, '#FFFFFF', 'CNB', 590)
    url = 'https://github.com/liangjianshu/multimodal-image-retrieval'
    demo_url = 'https://liangjianshu.github.io/multimodal-image-retrieval/overview/'
    for label, link, y in (
        ('GitHub 仓库 · 代码与文档', url, 855),
        ('在线演示 · 无需安装', demo_url, 913),
    ):
        p.text(label, 55, y, 16, YELLOW, 'CNB')
        p.text(link, 55, y+27, 11.5, '#FFFFFF', 'Helvetica', 475)
        width = pdfmetrics.stringWidth(link, 'Helvetica', 11.5)
        c.setStrokeColor(HexColor('#CED6D6'))
        c.setLineWidth(0.5)
        c.line(55,H-y-41,55+width,H-y-41)
        c.linkURL(link,(51,H-y-45,55+width+4,H-y+3),relative=0)
    p.text('PDF 内可点击链接；图片请扫码打开。', 55, 975, 12, '#CED6D6')
    p.rect(543, 851, 118, 118, '#FFFFFF', 5)
    qr = QrCodeWidget(url)
    x0,y0,x1,y1 = qr.getBounds()
    drawing = Drawing(108,108,transform=[108/(x1-x0),0,0,108/(y1-y0),0,0])
    drawing.add(qr)
    renderPDF.draw(drawing,c,548,H-856-108)
    p.text('扫码进入 GitHub', 550, 975, 12, '#CED6D6')
    c.linkURL(url,(543,H-969,661,H-851),relative=0)
    c.showPage()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', required=True)
    parser.add_argument('--bold-font', required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'docs/social/learning-pages.pdf')
    args = parser.parse_args()
    pdfmetrics.registerFont(TTFont('CN',args.font,subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CNB',args.bold_font,subfontIndex=0))
    metrics, hashes = data()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    c = canvas.Canvas(str(args.output),pagesize=(W,H),invariant=1)
    c.setTitle('多模态检索实战：实验结果与上手指南')
    c.setAuthor('liangjianshu')
    result_page(c,metrics)
    start_page(c)
    c.save()
    manifest={'source_sha256':hashes,'displayed_percentages':metrics,
              'qwen_protocol':'post-hoc normalized; validation-calibrated; not a new blind test',
              'embedding_protocol':'original frozen protocol',
              'scope':'primary five category mean retrieval F1; 2000-image gallery; single seed',
              'pages':['03 results','04 quickstart']}
    args.output.with_suffix('.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(args.output)


if __name__ == '__main__':
    main()
