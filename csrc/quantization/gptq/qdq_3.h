/*
 * Adapted from https://github.com/turboderp/exllamav2
 * SYCL/XPU port of qdq_3.cuh
 */

#pragma once
#include <cstdint>

namespace vllm {
namespace gptq {

// Permutation:
//
// v9997775 55333111  u8886664 44222000  (u, v lsb)
// vjjjhhhf ffdddbbb  uiiiggge eecccaaa
// vtttrrrp ppnnnlll  usssqqqo oommmkkk

inline void shuffle_3bit_32(uint32_t* q, int stride) {
  uint32_t qa = q[0 * stride];
  uint32_t qb = q[1 * stride];
  uint32_t qc = q[2 * stride];

  // qa: aa999888 77766655  54443332 22111000
  // qb: lkkkjjji iihhhggg  fffeeedd dcccbbba
  // qc: vvvuuutt tsssrrrq  qqpppooo nnnmmmll

  uint32_t qd = qc >> 26;
  qc <<= 4;
  qc |= qb >> 28;
  qb <<= 2;
  qb |= qa >> 30;

  // qa: ..999888 77766655  54443332 22111000
  // qb: ..jjjiii hhhgggff  feeedddc ccbbbaaa
  // qc: ..tttsss rrrqqqpp  pooonnnm mmlllkkk
  // qd:                               vvvuuu

  uint32_t za = 0;
  uint32_t zb = 0;
  uint32_t zc = 0;

  for (int i = 0; i < 5; i++) {
    uint32_t t0 = qa & 0x07;
    uint32_t t1 = (qa & 0x38) >> 3;
    qa >>= 6;
    za |= (t0 << (i * 3));
    za |= (t1 << (i * 3 + 16));
  }
  for (int i = 0; i < 5; i++) {
    uint32_t t0 = qb & 0x07;
    uint32_t t1 = (qb & 0x38) >> 3;
    qb >>= 6;
    zb |= (t0 << (i * 3));
    zb |= (t1 << (i * 3 + 16));
  }
  for (int i = 0; i < 5; i++) {
    uint32_t t0 = qc & 0x07;
    uint32_t t1 = (qc & 0x38) >> 3;
    qc >>= 6;
    zc |= (t0 << (i * 3));
    zc |= (t1 << (i * 3 + 16));
  }

  // za:  9997775 55333111   8886664 44222000
  // zb:  jjjhhhf ffdddbbb   iiiggge eecccaaa
  // zc:  tttrrrp ppnnnlll   sssqqqo oommmkkk
  // qd:                               vvvuuu

  za |= ((qd & 0x01) >> 0) << 15;
  zb |= ((qd & 0x02) >> 1) << 15;
  zc |= ((qd & 0x04) >> 2) << 15;
  za |= ((qd & 0x08) >> 3) << 31;
  zb |= ((qd & 0x10) >> 4) << 31;
  zc |= ((qd & 0x20) >> 5) << 31;

  // za: v9997775 55333111  u8886664 44222000  (u, v lsb)
  // zb: vjjjhhhf ffdddbbb  uiiiggge eecccaaa
  // zc: vtttrrrp ppnnnlll  usssqqqo oommmkkk

  q[0 * stride] = za;
  q[1 * stride] = zb;
  q[2 * stride] = zc;
}

}  // namespace gptq
}  // namespace vllm
