import sys, os
os.environ['MPLCONFIGDIR'] = 'C:/Temp/mpl'
sys.path.insert(0, '.')
from src.inference.pipeline import HistoricImageRestorationPipeline, RestorationConfig
import cv2, numpy as np
from pathlib import Path

pipe = HistoricImageRestorationPipeline()
demo_dir = Path('demo_images')

for label, idx in [('Hiker', 1), ('Desert', 3), ('Night', 2)]:
    img = cv2.imread(str(demo_dir / f'user_damaged_{idx}.png'))
    if img is None:
        print(f'Missing user_damaged_{idx}.png')
        continue
    
    mask = pipe.detect_defects(img, detect_bright=True, detect_dark=False, sensitivity=60, dilation_radius=2)
    pct = np.sum(mask > 0) / mask.size * 100
    n, labels_arr, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    
    print(f'{label} mask: {np.sum(mask>0)} px = {pct:.2f}% | {n-1} components')
    
    cfg = RestorationConfig()
    result = pipe.restore(img, cfg)
    
    out = demo_dir / f'user_restored_{idx}_v2.png'
    cv2.imwrite(str(out), result['restored_bgr'])
    
    t = result['metrics']['processing_time_s']
    s_in = result['metrics']['sharpness_original']
    s_out = result['metrics']['sharpness_restored']
    print(f'  Time: {t:.2f}s  Sharpness: {s_in:.1f} -> {s_out:.1f}')

print('Done.')
