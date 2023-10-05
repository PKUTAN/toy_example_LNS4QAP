\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  - 157.892 x25 - 215.687 x24 - 234.345 x23 - 209.091 x22 - 150.544 x21
   - 198.286 x20 - 191.853 x19 - 157.5 x18 - 100.155 x17 - 206.873 x16
   - 107.749 x15 - 185.207 x14 - 130.151 x13 - 74.2941 x12 - 125.471 x11
   - 165.135 x10 - 191.339 x9 - 341.932 x8 - 72.185 x7 - 153.35 x6
   - 194.624 x5 - 98.4412 x4 - 323.057 x3 - 203.626 x2 - 21.5457 x1
   - 256.977 x0
Subject To
 c0: x24 + x23 + x22 + x20 + x19 + x17 + x16 + x15 + x9 + x8 + x7 + x5 + x3
   + x2 + x0 <= 1
 c1: x25 + x24 + x23 + x22 + x21 + x20 + x19 + x18 + x16 + x14 + x13 + x11
   + x10 + x9 + x8 + x3 + x2 + x0 <= 1
 c2: x25 + x24 + x22 + x21 + x19 + x18 + x17 + x15 + x14 + x13 + x8 + x6
   + x5 + x3 + x2 + x1 <= 1
 c3: x25 + x23 + x21 + x20 + x18 + x17 + x16 + x14 + x13 + x12 + x10 + x8
   + x6 + x5 + x3 + x0 <= 1
 c36: x25 + x24 <= 1
 c34: x21 + x20 + x17 <= 1
 c35: x23 + x22 <= 1
 c24: x13 + x6 <= 1
 c4: x14 + x8 + x6 + x4 + x2 + x0 <= 1
 c31: x18 + x16 <= 1
Bounds
Binaries
 x25 x24 x23 x22 x21 x20 x19 x18 x17 x16 x15 x14 x13 x12 x11 x10 x9 x8 x7
 x6 x5 x4 x3 x2 x1 x0
End
