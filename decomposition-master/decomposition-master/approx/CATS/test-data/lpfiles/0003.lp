\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  - 264.027 x24 - 114.405 x23 - 228.277 x22 - 226.76 x21 - 224.68 x20
   - 15.5517 x19 - 142.838 x18 - 402.118 x17 - 345.71 x16 - 87.4289 x15
   - 62.1876 x14 - 176.658 x13 - 231.855 x12 - 160.972 x11 - 307 x10
   - 230.85 x9 - 192.541 x8 - 133.771 x7 - 209.54 x6 - 192.221 x5
   - 104.18 x4 - 198.672 x3 - 111.335 x2 - 391.95 x1 - 327.749 x0
Subject To
 c0: x24 + x22 + x21 + x20 + x17 + x16 + x13 + x12 + x11 + x10 + x8 + x7
   + x6 + x3 + x1 + x0 <= 1
 c1: x24 + x23 + x22 + x19 + x16 + x15 + x13 + x11 + x10 + x9 + x8 + x3
   + x1 + x0 <= 1
 c2: x23 + x22 + x21 + x20 + x18 + x17 + x16 + x13 + x12 + x11 + x10 + x9
   + x6 + x5 + x2 + x1 + x0 <= 1
 c3: x24 + x23 + x21 + x18 + x17 + x16 + x15 + x14 + x9 + x6 + x5 + x3 + x1
   + x0 <= 1
 c4: x24 + x17 + x12 + x10 + x9 + x5 + x4 + x1 <= 1
 c8: x13 + x12 <= 1
 c18: x17 + x16 <= 1
 c20: x20 + x8 <= 1
 c22: x23 + x22 + x21 <= 1
 c23: x24 <= 1
Bounds
Binaries
 x24 x23 x22 x21 x20 x19 x18 x17 x16 x15 x14 x13 x12 x11 x10 x9 x8 x7 x6 x5
 x4 x3 x2 x1 x0
End
