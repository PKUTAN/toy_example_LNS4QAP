\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  - 117.052 x24 - 148.217 x23 - 241.466 x22 - 325.832 x21 - 275.538 x20
   - 312.117 x19 - 281.899 x18 - 289.432 x17 - 239.177 x16 - 307.125 x15
   - 390.339 x14 - 129.926 x13 - 210.221 x12 - 272.267 x11 - 250.149 x10
   - 242.237 x9 - 386.276 x8 - 130.285 x7 - 89.5809 x6 - 80.7494 x5
   - 478.726 x4 - 234.391 x3 - 267.737 x2 - 459.653 x1 - 323.029 x0
Subject To
 c0: x22 + x21 + x20 + x18 + x17 + x15 + x14 + x13 + x12 + x11 + x8 + x4
   + x2 + x1 + x0 <= 1
 c1: x22 + x21 + x20 + x19 + x18 + x16 + x15 + x14 + x11 + x10 + x9 + x8
   + x7 + x4 + x3 + x1 <= 1
 c2: x23 + x22 + x19 + x17 + x16 + x15 + x14 + x12 + x10 + x9 + x5 + x4
   + x2 + x1 + x0 <= 1
 c3: x24 + x23 + x21 + x19 + x18 + x17 + x16 + x10 + x9 + x8 + x4 + x3 + x2
   + x1 + x0 <= 1
 c4: x20 + x19 + x14 + x6 + x4 + x0 <= 1
 c6: x9 + x8 <= 1
 c12: x10 + x2 <= 1
 c19: x17 + x16 <= 1
 c22: x22 + x21 + x20 <= 1
Bounds
Binaries
 x24 x23 x22 x21 x20 x19 x18 x17 x16 x15 x14 x13 x12 x11 x10 x9 x8 x7 x6 x5
 x4 x3 x2 x1 x0
End
