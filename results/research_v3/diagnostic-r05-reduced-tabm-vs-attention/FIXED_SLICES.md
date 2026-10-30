# All fixed descriptive train-OOF slices

Reduced / Full / Attention denote the three fixed methods in paired-and-slices.json. No additional split or threshold is fitted.

| Dimension | Bucket | N | Reduced correct | Full correct | Attention correct | Reduced - full | Reduced-only vs Attention | Attention-only vs reduced | Both wrong |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| answer_groups | 1 | 2641 | 2641 | 2641 | 2641 | 0 | 0 | 0 | 0 |
| answer_groups | 2 | 1338 | 1143 | 1113 | 1151 | 30 | 20 | 28 | 167 |
| answer_groups | 3 | 608 | 428 | 403 | 431 | 25 | 17 | 20 | 160 |
| answer_groups | 4 | 168 | 97 | 90 | 103 | 7 | 6 | 12 | 59 |
| positive_candidates | 1 | 711 | 364 | 322 | 388 | 42 | 20 | 44 | 303 |
| positive_candidates | 2 | 732 | 657 | 637 | 652 | 20 | 14 | 9 | 66 |
| positive_candidates | 3 | 1070 | 1046 | 1046 | 1044 | 0 | 9 | 7 | 17 |
| positive_candidates | 4 | 2242 | 2242 | 2242 | 2242 | 0 | 0 | 0 | 0 |
| unparseable_candidates | 0 | 4019 | 3659 | 3609 | 3672 | 50 | 35 | 48 | 312 |
| unparseable_candidates | 1 | 628 | 550 | 541 | 556 | 9 | 6 | 12 | 66 |
| unparseable_candidates | 2 | 81 | 73 | 70 | 71 | 3 | 2 | 0 | 8 |
| unparseable_candidates | 3 | 27 | 27 | 27 | 27 | 0 | 0 | 0 | 0 |
| unparseable_candidates | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| question_words | [0,40) | 2359 | 2203 | 2183 | 2202 | 20 | 20 | 19 | 137 |
| question_words | [40,80) | 2244 | 1974 | 1935 | 1994 | 39 | 19 | 39 | 231 |
| question_words | [80,120) | 144 | 125 | 121 | 123 | 4 | 4 | 2 | 17 |
| question_words | [120,inf) | 8 | 7 | 8 | 7 | -1 | 0 | 0 | 1 |
| question_characters | [0,200) | 2235 | 2092 | 2073 | 2092 | 19 | 17 | 17 | 126 |
| question_characters | [200,400) | 2324 | 2050 | 2013 | 2066 | 37 | 22 | 38 | 236 |
| question_characters | [400,600) | 184 | 158 | 150 | 159 | 8 | 4 | 5 | 21 |
| question_characters | [600,inf) | 12 | 9 | 11 | 9 | -2 | 0 | 0 | 3 |
| candidate_mean_words | [0,100) | 2911 | 2702 | 2671 | 2707 | 31 | 21 | 26 | 183 |
| candidate_mean_words | [100,200) | 1800 | 1572 | 1542 | 1582 | 30 | 22 | 32 | 196 |
| candidate_mean_words | [200,400) | 44 | 35 | 34 | 37 | 1 | 0 | 2 | 7 |
| candidate_mean_words | [400,inf) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| candidate_max_words | [0,100) | 2125 | 1995 | 1978 | 1995 | 17 | 15 | 15 | 115 |
| candidate_max_words | [100,200) | 2385 | 2115 | 2078 | 2127 | 37 | 23 | 35 | 235 |
| candidate_max_words | [200,400) | 245 | 199 | 191 | 204 | 8 | 5 | 10 | 36 |
| candidate_max_words | [400,inf) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| candidate_word_spread | [0,50) | 3635 | 3362 | 3325 | 3367 | 37 | 28 | 33 | 240 |
| candidate_word_spread | [50,100) | 969 | 825 | 807 | 832 | 18 | 12 | 19 | 125 |
| candidate_word_spread | [100,200) | 145 | 117 | 109 | 122 | 8 | 3 | 8 | 20 |
| candidate_word_spread | [200,inf) | 6 | 5 | 6 | 5 | -1 | 0 | 0 | 1 |
| candidate_mean_characters | [0,500) | 2839 | 2637 | 2611 | 2637 | 26 | 23 | 23 | 179 |
| candidate_mean_characters | [500,1000) | 1877 | 1642 | 1607 | 1658 | 35 | 20 | 36 | 199 |
| candidate_mean_characters | [1000,2000) | 39 | 30 | 29 | 31 | 1 | 0 | 1 | 8 |
| candidate_mean_characters | [2000,inf) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
