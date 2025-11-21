import sys
sys.path.append('/home/caoren/tmp/PIRC_for_HigherOrderCausality')
from Torch_Library import *

cols = 5
ExpandNodes = cols + math.comb(cols - 1, 2)
device = "cpu"
w1 = generate_structured_w(cols, ExpandNodes, device, 1)
w2 = generate_structured_w(cols, ExpandNodes, device, 2)
w3= generate_structured_w(cols, ExpandNodes*2, device)
print("structured_w", w)