# untrained 
baseline: 11.495843970775605
intensity + masking: 3.546254646778107
intensity + no masking: 4.148271602392197
MIND + masking: 2.305850052833557
MIND + no masking: 3.0112798511981964

# trained net:

masked_res_gn_gelu_resgn_32h_8out (learned): 1.895914(train best @ gen 40) 1.865654(test @ train-best gen) 1.820674(test best @ gen 34)
masked_res_gn_gelu_resgn_cascade2 (learned): 1.944174(train best @ gen 40) 2.024269(test @ train-best gen) 1.893234(test best @ gen 3)
masked_res_gn_gelu (learned): 1.912820(train best @ gen 39) 2.103606(test @ train-best gen) 1.893980(test best @ gen 25) 
masked_res_gn_gelu_resgn_mindconcat (mind_concat): 2.060511(train best @ gen 26) 2.241511(test @ train-best gen) 2.203031(test best @ gen 8)
masked_res_gn_gelu_resgn_dil2 (learned): 2.316168(train best @ gen 37) 2.291432(test @ train-best gen) 2.291432(test best @ gen 37)

masked_cbam (learned): 2.242989(train best @ gen 40) 2.191270(test @ train-best gen) 2.154270(test best @ gen 39)


masked_basic (learned): 2.198864(train best @ gen 43) 2.422595(test @ train-best gen) 2.374192(test best @ gen 24)
unmasked_basic (learned): 3.138428(train best @ gen 43) 3.541092(test @ train-best gen) 2.848910(test best @ gen 8)

