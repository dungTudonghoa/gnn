"""Render a report using measured runs only; no paper results are substituted."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(directory):
    root = Path(directory)
    stats = json.loads((root/'statistics.json').read_text())
    training = json.loads((root/'training.json').read_text())
    rows = json.loads((root/'runs.json').read_text())
    environment = json.loads((root/'environment.json').read_text())
    best = training['history'][training['best_epoch']-1]
    groups = sorted({(r['instance'],r['formulation'],r['budget']) for r in rows})
    text = ['# Kết quả chạy thực tế: bản triển khai độc lập', '',
            '**Các số dưới đây tính từ dữ liệu dựng lại, không phải kết quả tái tạo Tables III–VII của tác giả.**', '',
            f"Đã chạy {len(rows)} lượt tìm kiếm, tổng {sum(r['evaluations'] for r in rows):,} LP cho assignment mới. "
            'Các lượt tight-budget dùng 20 seed; sensitivity dùng 10 seed. Reference được giải bằng MILP đến optimal.', '',
            f"GNN: chọn epoch {training['best_epoch']}, validation accuracy {100*best['val_accuracy']:.2f}%; "
            f"train {len(training['history'])} epoch trước khi dừng, thời gian train {training['training_seconds']:.2f} giây. "
            'Accuracy này đo trên ba fixture validation tự dựng, không so với 75.56% của bài như cùng một test.', '',
            '| Fixture | Bài toán | Budget | Exact reference | SA mean ± SD | GA mean ± SD | GNN–GA mean ± SD |',
            '|---|---|---:|---:|---:|---:|---:|']
    for name,form,budget in groups:
        subset = {r['method']:r for r in stats['summary'] if (r['instance'],r['formulation'],r['budget'])==(name,form,budget)}
        def fmt(method):
            if method not in subset: return '—'
            r = subset[method]; return f"{r['mean']:,.2f} ± {r['sd']:,.2f}"
        text.append(f"| {name} | {form} | {budget} | {subset['ga']['reference']:,.2f} | {fmt('sa')} | {fmt('ga')} | {fmt('gnn_ga')} |")
    text += ['', '## Diễn giải trong phạm vi lần chạy', '',
             '- Tight budgets 75/25/10 nhỏ hơn population 100: GA/GNN–GA chỉ khởi tạo, không có thế hệ con hoàn chỉnh.',
             '- Sensitivity B=400 trên fixture test-01 chạy ba thế hệ hoàn chỉnh và một thế hệ dở dang. Đây không phải Instance 13 của bài.',
             '- `initialization_only` và `gnn_ga` có cùng kết quả theo seed ở tight budget vì mutation chưa được dùng.',
             f"- Có tổng {sum(r['fallback_count'] for r in rows)} lần dùng chính sách tìm chromosome chưa thấy sau nhiều duplicate; chính sách này được công khai trong REPRODUCIBILITY.md.",
             '- Số liệu cho thấy lợi ích trên các fixture này; không đủ để xác nhận các tỷ lệ tốc độ, p-value hay mức cải thiện của dữ liệu gốc.', '',
             '![Measured objectives](comparison.png)', '',
             '## Tệp kiểm toán', '',
             '- `manifest.json`, `instances/`: toàn bộ dữ liệu đầu vào và split.',
             '- `model.pt`, `training.json`, `labels.json`: checkpoint, normalizer, lịch sử train và nhãn tối ưu.',
             '- `references.json`: optimum theo scenario, nominal/regret và assignment.',
             '- `runs.json`: objective, assignment, seed, budget, số LP, số thế hệ, fallback và thời gian từng run.',
             '- `statistics.json`, `summary.csv`: mean, sample SD, success count, deviation/gap và thống kê ghép cặp.',
             '- `environment.json`: phiên bản runtime và SHA-256 từng instance.', '',
             '## Môi trường', '',
             f"Python {environment['python']}, PyTorch {environment['torch']}, NumPy {environment['numpy']}, SciPy {environment['scipy']}; CPU, một thread.", '',
             'Nguồn phương pháp: [arXiv:2608.10245v1](https://arxiv.org/abs/2608.10245). '
             'Mức độ khớp và lựa chọn chưa được tác giả công bố: [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md).', '']
    (root/'REPORT.md').write_text('\n'.join(text))
    fig,axes = plt.subplots(2,3,figsize=(13,7),layout='constrained')
    methods = ['sa','ga','gnn_ga','cost']
    names = sorted({r['instance'] for r in rows})
    for column,name in enumerate(names[:3]):
        for row,form in enumerate(['deterministic','regret']):
            subset = {r['method']:r for r in stats['summary'] if r['instance']==name and r['formulation']==form and r['budget']!=400}
            ax = axes[row,column]
            means = [subset[m]['mean'] for m in methods]; stds = [subset[m]['sd'] for m in methods]
            ax.bar(['SA','GA','GNN-GA','Cost'],means,yerr=stds,capsize=3,color=['#9ba8b7','#6385ad','#228c77','#b49465'])
            ax.axhline(subset['ga']['reference'],color='#ae3844',linestyle='--',label='Exact reference')
            ax.set_title(name.replace('reconstructed-','')+' / '+form)
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            ax.set_ylabel('Objective (lower is better)')
            if row==0 and column==0: ax.legend(fontsize=8)
    fig.suptitle('Independent reconstructed data — measured mean ± sample SD, 20 seeds',fontsize=14)
    fig.savefig(root/'comparison.png',dpi=160); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('directory',nargs='?',default='supply_chain/results/reconstruction')
    render(parser.parse_args().directory)
