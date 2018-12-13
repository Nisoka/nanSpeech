/* Copyright (c) 2017 Mozilla */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include "common.h"
#include <math.h>
#include "rnnoise.h"
#include "pitch.h"
#include "arch.h"
#include "rnn.h"
#include "rnn_data.h"

#define STB_FFT_IMPLEMENTAION

#include "stb_fft.h"

#define FRAME_SIZE_SHIFT 2
#define FRAME_SIZE (120<<FRAME_SIZE_SHIFT)
#define WINDOW_SIZE (2*FRAME_SIZE)
#define FREQ_SIZE (FRAME_SIZE + 1)

#define PITCH_MIN_PERIOD 60
#define PITCH_MAX_PERIOD 768
#define PITCH_FRAME_SIZE 960
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD+PITCH_FRAME_SIZE)

#define SQUARE(x) ((x)*(x))

#define SMOOTH_BANDS 1

#if SMOOTH_BANDS
#define NB_BANDS 22
#else
#define NB_BANDS 21
#endif

#define CEPS_MEM 8
#define NB_DELTA_CEPS 6

#define NB_FEATURES (NB_BANDS+3*NB_DELTA_CEPS+2)


#define TRAINING 0

static const opus_int16 eband5ms[] = {
/*0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k 9.6 12k 15.6 20k*/
  0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
};

typedef struct {
    int init;
    stb_fft_real_plan *kfft;
    float half_window[FRAME_SIZE];
    float dct_table[NB_BANDS * NB_BANDS];
} CommonState;

struct DenoiseState {
    float analysis_mem[FRAME_SIZE];
    float cepstral_mem[CEPS_MEM][NB_BANDS];
    int memid;
    float synthesis_mem[WINDOW_SIZE];
    float pitch_buf[PITCH_BUF_SIZE];
    float last_gain;
    int last_period;
    float mem_hp_x[2];
    float lastg[NB_BANDS];
    RNNState rnn;
};

#if SMOOTH_BANDS
void compute_band_energy(float *bandE, const cmplx *X) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].real);
      tmp += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].imag);
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
  }
  sum[0] *= 2;
  sum[NB_BANDS-1] *= 2;
    memcpy(bandE, sum, NB_BANDS * sizeof(float));
}

void compute_band_corr(float *bandE, const cmplx *X, const cmplx *P) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].real * P[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].real;
      tmp += X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].imag * P[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].imag;
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
  }
  sum[0] *= 2;
  sum[NB_BANDS-1] *= 2;
    memcpy(bandE, sum, NB_BANDS * sizeof(float));
}

void interp_band_gain(float *g, const float *bandE) {
  int i;
  memset(g, 0, FREQ_SIZE);
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    for (j=0;j<band_size;j++) {
      float frac = (float)j/band_size;
      g[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j] = (1-frac)*bandE[i] + frac*bandE[i+1];
    }
  }
}
#else
void compute_band_energy(float *bandE, const cmplx *X) {
  int i;
  for (i=0;i<NB_BANDS;i++)
  {
    int j;
    opus_val32 sum = 0;
    for (j=0;j<(eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;j++) {
      sum += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r);
      sum += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i);
    }
    bandE[i] = sum;
  }
}

void interp_band_gain(float *g, const float *bandE) {
  int i;
  memset(g, 0, FREQ_SIZE);
  for (i=0;i<NB_BANDS;i++)
  {
    int j;
    for (j=0;j<(eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;j++)
      g[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j] = bandE[i];
  }
}
#endif


CommonState common;

static void check_init() {
    int i;
    if (common.init) return;
    int plan_size = stb_fft_real_plan_dft_1d(WINDOW_SIZE, NULL);
    if (plan_size <= 0)
        return;
    common.kfft = (stb_fft_real_plan *) calloc(plan_size, 1);
    stb_fft_real_plan_dft_1d(WINDOW_SIZE, common.kfft);
    for (i = 0; i < FRAME_SIZE; i++)
        common.half_window[i] = (float) sin(
                .5 * M_PI * sin(.5 * M_PI * (i + .5) / FRAME_SIZE) * sin(.5 * M_PI * (i + .5) / FRAME_SIZE));
    float sq = sqrtf(.5);
    for (i = 0; i < NB_BANDS; i++) {
        int j;
        for (j = 0; j < NB_BANDS; j++) {
            common.dct_table[i * NB_BANDS + j] = (float) cos((i + .5) * j * M_PI / NB_BANDS);
            if (j == 0) common.dct_table[i * NB_BANDS + j] *= sq;
        }
    }
    common.init = 1;
}

static void dct(float *out, const float *in) {
    int i;
    check_init();
    float sq = sqrtf(2.f / 22);
    for (i = 0; i < NB_BANDS; i++) {
        int j;
        float sum = 0;
        for (j = 0; j < NB_BANDS; j++) {
            sum += in[j] * common.dct_table[j * NB_BANDS + i];
        }
        out[i] = sum * sq;
    }
}

#if 0
static void idct(float *out, const float *in) {
  int i;
  check_init();
    float sq = sqrtf(2.f / 22);
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * common.dct_table[i*NB_BANDS + j];
    }
    out[i] = sum*sq;
  }
}
#endif

static void forward_transform(cmplx *out, float *in) {
    int i;
    check_init();
    float norm = 1.0f / WINDOW_SIZE;
    for (i = 0; i < FRAME_SIZE; i++) {
        const float t = common.half_window[i] * norm;
        in[i] *= t;
        in[WINDOW_SIZE - 1 - i] *= t;
    }
    stb_fft_r2c_exec(common.kfft, in, out);
}


size_t rnnoise_get_size() {
    return sizeof(DenoiseState);
}

int rnnoise_init(DenoiseState *st) {
  memset(st, 0, sizeof(*st));
  return 0;
}

DenoiseState *rnnoise_create() {
    DenoiseState *st = rnnoise_alloc(rnnoise_get_size());
    if (st == NULL) {
        printf("[%s %d] malloc failed\n", __FUNCTION__, __LINE__);
        return NULL;
    }
    rnnoise_init(st);
    return st;
}

void rnnoise_destroy(DenoiseState *st) {
if (st)
    free(st);
}

#if TRAINING
int lowpass = FREQ_SIZE;
int band_lp = NB_BANDS;
#endif

static void frame_analysis(DenoiseState *st, cmplx *X, float *Ex, const float *in) {
    float *x = rnnoise_alloc(WINDOW_SIZE * sizeof(float));
    if (x == NULL) {
        printf("[%s %d] malloc failed\n", __FUNCTION__, __LINE__);
        return;
    }
    memcpy(x, st->analysis_mem, sizeof(float) * FRAME_SIZE);
    memcpy(x + FRAME_SIZE, in, sizeof(float) * FRAME_SIZE);
    memcpy(st->analysis_mem, in, sizeof(float) * FRAME_SIZE);
    forward_transform(X, x);
#if TRAINING
    for (i = lowpass; i < FREQ_SIZE; i++)
        X[i].r = X[i].i = 0;
#endif
    compute_band_energy(Ex, X);
    rnnoise_free(x);
}

static int compute_frame_features(DenoiseState *st, cmplx *X, cmplx *P,
                                  float *Ex, float *Ep, float *Exp, float *features, const float *in) {
    int i;
    float E = 0;
    float *ceps_0, *ceps_1, *ceps_2;
    float spec_variability = 0;
    float Ly[NB_BANDS];
    float *p = rnnoise_alloc(WINDOW_SIZE * sizeof(float));
    if (p == NULL) {
        printf("[%s %d] malloc failed\n", __FUNCTION__, __LINE__);
        return -1;
    }
    float pitch_buf[PITCH_BUF_SIZE >> 1];
    int pitch_index;
    float gain;
    float *(pre[1]);
    float tmp[NB_BANDS];
    float follow, logMax;
    frame_analysis(st, X, Ex, in);
    memmove(st->pitch_buf, st->pitch_buf + FRAME_SIZE, (PITCH_BUF_SIZE - FRAME_SIZE) * sizeof(*st->pitch_buf));
    memcpy(&st->pitch_buf[PITCH_BUF_SIZE - FRAME_SIZE], in, FRAME_SIZE * sizeof(*st->pitch_buf));
    pre[0] = &st->pitch_buf[0];
    pitch_downsample(pre, pitch_buf, PITCH_BUF_SIZE, 1);
    pitch_search(pitch_buf + (PITCH_MAX_PERIOD >> 1), pitch_buf, PITCH_FRAME_SIZE,
                 PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD, &pitch_index);
    pitch_index = PITCH_MAX_PERIOD - pitch_index;

    gain = remove_doubling(pitch_buf, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD,
                           PITCH_FRAME_SIZE, &pitch_index, st->last_period, st->last_gain);
    st->last_period = pitch_index;
    st->last_gain = gain;
    memcpy(p, st->pitch_buf + PITCH_BUF_SIZE - WINDOW_SIZE - pitch_index, sizeof(float) * WINDOW_SIZE);
    forward_transform(P, p);
    compute_band_energy(Ep, P);
    compute_band_corr(Exp, X, P);
    for (i = 0; i < NB_BANDS; i++)
        Exp[i] = Exp[i] * fastInvSqrt(.001f + Ex[i] * Ep[i]);
    dct(tmp, Exp);
    memcpy(features + NB_BANDS + 2 * NB_DELTA_CEPS, tmp, sizeof(float) * NB_DELTA_CEPS);
    features[NB_BANDS + 2 * NB_DELTA_CEPS] -= 1.3f;
    features[NB_BANDS + 2 * NB_DELTA_CEPS + 1] -= 0.9f;
    features[NB_BANDS + 3 * NB_DELTA_CEPS] = .01f * (pitch_index - 300);
    logMax = -2;
    follow = -2;
    for (i = 0; i < NB_BANDS; i++) {
        Ly[i] = log10f(1e-2f + Ex[i]);
        Ly[i] = MAX16(logMax - 7, MAX16(follow - 1.5f, Ly[i]));
        logMax = MAX16(logMax, Ly[i]);
        follow = MAX16(follow - 1.5f, Ly[i]);
        E += Ex[i];
    }
    if (!TRAINING && E < 0.04f) {
        /* If there's no audio, avoid messing up the state. */
        memset(features, 0, (NB_FEATURES) * sizeof(*features));
        rnnoise_free(p);
        return 1;
    }
    dct(features, Ly);
    features[0] -= 12;
    features[1] -= 4;
    ceps_0 = st->cepstral_mem[st->memid];
    ceps_1 = (st->memid < 1) ? st->cepstral_mem[CEPS_MEM + st->memid - 1] : st->cepstral_mem[st->memid - 1];
    ceps_2 = (st->memid < 2) ? st->cepstral_mem[CEPS_MEM + st->memid - 2] : st->cepstral_mem[st->memid - 2];
    memcpy(ceps_0, features, sizeof(float) * NB_BANDS);
    st->memid++;
    for (i = 0; i < NB_DELTA_CEPS; i++) {
        features[i] = ceps_0[i] + ceps_2[i];
        features[NB_BANDS + i] = ceps_0[i] - ceps_2[i];
        features[NB_BANDS + NB_DELTA_CEPS + i] = features[i] - 2 * ceps_1[i];
        features[i] += ceps_1[i];
    }
    /* Spectral variability features. */
    if (st->memid == CEPS_MEM) st->memid = 0;
    for (i = 0; i < CEPS_MEM; i++) {
        int j;
        float mindist = 1e15f;
        for (j = 0; j < CEPS_MEM; j++) {
            int k;
            float dist = 0;
            for (k = 0; k < NB_BANDS; k++) {
                float tmp;
                tmp = st->cepstral_mem[i][k] - st->cepstral_mem[j][k];
                dist += tmp * tmp;
            }
            if (j != i)
                mindist = MIN32(mindist, dist);
        }
        spec_variability += mindist;
    }
    features[NB_BANDS + 3 * NB_DELTA_CEPS + 1] = spec_variability / CEPS_MEM - 2.1f;
    rnnoise_free(p);
    return TRAINING && E < 0.1f;
}

static void frame_synthesis(DenoiseState *st, short *out, cmplx *input) {
    check_init();
    int i;
    int idx = 0;
    for (i = FREQ_SIZE; i < WINDOW_SIZE; i++) {
        input[i].real = input[WINDOW_SIZE - i].real;
        input[i].imag = -input[WINDOW_SIZE - i].imag;
        idx = i - FREQ_SIZE;
        out[idx] = (short) (st->synthesis_mem[idx + FRAME_SIZE]);
    }
    stb_fft_c2r_exec(common.kfft, input, st->synthesis_mem);
    for (i = 0; i < FRAME_SIZE; i++) {
        out[i] += (short) (st->synthesis_mem[i] * common.half_window[i]);
        st->synthesis_mem[WINDOW_SIZE - 1 - i] *= common.half_window[i];
    }
}

static void biquad(float *y, float mem[2], const short *x, const float *b, const float *a, int N) {
    int i;
    for (i = 0; i < N; i++) {
        float xi, yi;
        xi = x[i];
        yi = x[i] + mem[0];
        mem[0] = mem[1] + (b[0] * xi - a[0] * yi);
        mem[1] = (b[1] * xi - a[1] * yi);
        y[i] = yi;
    }
}

void pitch_filter(cmplx *X, const cmplx *P, const float *Ex, const float *Ep,
                  const float *Exp, const float *g) {
    int i;
    float r[NB_BANDS];
    float *rf = rnnoise_alloc(FREQ_SIZE * sizeof(float));
    if (rf == NULL) {
        printf("[%s %d] malloc failed\n", __FUNCTION__, __LINE__);
        return;
    }
    memset(rf, 0, FREQ_SIZE * sizeof(float));

    for (i = 0; i < NB_BANDS; i++) {
#if 0
        if (Exp[i]>g[i]) r[i] = 1;
        else r[i] = Exp[i]*(1-g[i])/(.001 + g[i]*(1-Exp[i]));
        r[i] = MIN16(1, MAX16(0, r[i]));
#else
        if (Exp[i] > g[i])
            r[i] = 1;
        else
            r[i] = SQUARE(Exp[i]) * (1 - SQUARE(g[i])) / (.001f + SQUARE(g[i]) * (1 - SQUARE(Exp[i])));
        r[i] = 1.f / fastInvSqrt(MIN16(1, MAX16(0, r[i])));
#endif
        r[i] *= 1.f / fastInvSqrt(Ex[i] / (1e-8f + Ep[i]));
    }
    interp_band_gain(rf, r);
    for (i = 0; i < FREQ_SIZE; i++) {
        X[i].real += rf[i] * P[i].real;
        X[i].imag += rf[i] * P[i].imag;
    }
    float newE[NB_BANDS];
    compute_band_energy(newE, X);
    float norm[NB_BANDS];
    float *normf = rnnoise_alloc(FREQ_SIZE * sizeof(float));
    if (normf == NULL) {
        printf("[%s %d] malloc failed\n", __FUNCTION__, __LINE__);
        rnnoise_free(rf);
        return;
    }
    for (i = 0; i < NB_BANDS; i++) {
        norm[i] =  1.f / fastInvSqrt(Ex[i] / (1e-8f + newE[i]));
    }
    interp_band_gain(normf, norm);
    for (i = 0; i < FREQ_SIZE; i++) {
        X[i].real *= normf[i];
        X[i].imag *= normf[i];
    }
    rnnoise_free(rf);
    rnnoise_free(normf);
}

float rnnoise_process_frame(DenoiseState *st, short *out, const short *in) {
    int i;
    cmplx *input = rnnoise_alloc(WINDOW_SIZE * sizeof(cmplx));
    cmplx *P = rnnoise_alloc(WINDOW_SIZE * sizeof(cmplx));
    if ((input == NULL) || (P == NULL)) {
        printf("[%s %d] malloc failed\n", __FUNCTION__, __LINE__);
        rnnoise_free(P);
        rnnoise_free(input);
        return 0;
    }
    float Ex[NB_BANDS], Ep[NB_BANDS];
    float Exp[NB_BANDS];
    float features[NB_FEATURES];
    float g[NB_BANDS];
    float gf[FREQ_SIZE] = {1};
    float vad_prob = 0;
    int silence;
    static const float a_hp[2] = {-1.99599f, 0.99600f};
    static const float b_hp[2] = {-2, 1};
    biquad(st->synthesis_mem, st->mem_hp_x, in, b_hp, a_hp, FRAME_SIZE);
    silence = compute_frame_features(st, input, P, Ex, Ep, Exp, features, st->synthesis_mem);
    if (!silence) {
        compute_rnn(&st->rnn, g, &vad_prob, features);
        pitch_filter(input, P, Ex, Ep, Exp, g);
        for (i = 0; i < NB_BANDS; i++) {
            float alpha = .6f;
            g[i] = MAX16(g[i], alpha * st->lastg[i]);
            st->lastg[i] = g[i];
        }
        interp_band_gain(gf, g);
#if 1
        for (i = 0; i < FREQ_SIZE; i++) {
            input[i].real *= gf[i];
            input[i].imag *= gf[i];
        }
#endif
    }
    frame_synthesis(st, out, input);
    rnnoise_free(input);
    rnnoise_free(P);
    return vad_prob;
}

#if TRAINING

static float uni_rand() {
  return rand()/(double)RAND_MAX-.5;
}

static void rand_resp(float *a, float *b) {
  a[0] = .75*uni_rand();
  a[1] = .75*uni_rand();
  b[0] = .75*uni_rand();
  b[1] = .75*uni_rand();
}

int main(int argc, char **argv) {
  int i;
  int count=0;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  float a_noise[2] = {0};
  float b_noise[2] = {0};
  float a_sig[2] = {0};
  float b_sig[2] = {0};
  float mem_hp_x[2]={0};
  float mem_hp_n[2]={0};
  float mem_resp_x[2]={0};
  float mem_resp_n[2]={0};
  float x[FRAME_SIZE];
  float n[FRAME_SIZE];
  float xn[FRAME_SIZE];
  int vad_cnt=0;
  int gain_change_count=0;
  float speech_gain = 1, noise_gain = 1;
  FILE *f1, *f2, *fout;
  DenoiseState *st;
  DenoiseState *noise_state;
  DenoiseState *noisy;
  st = rnnoise_create();
  noise_state = rnnoise_create();
  noisy = rnnoise_create();
  if (argc!=4) {
    fprintf(stderr, "usage: %s <speech> <noise> <output denoised>\n", argv[0]);
    return 1;
  }
  f1 = fopen(argv[1], "r");
  f2 = fopen(argv[2], "r");
  fout = fopen(argv[3], "w");
  for(i=0;i<150;i++) {
    short tmp[FRAME_SIZE];
    fread(tmp, sizeof(short), FRAME_SIZE, f2);
  }
  while (1) {
    kiss_fft_cpx X[FREQ_SIZE], Y[FREQ_SIZE], N[FREQ_SIZE], P[WINDOW_SIZE];
    float Ex[NB_BANDS], Ey[NB_BANDS], En[NB_BANDS], Ep[NB_BANDS];
    float Exp[NB_BANDS];
    float Ln[NB_BANDS];
    float features[NB_FEATURES];
    float g[NB_BANDS];
    float gf[FREQ_SIZE]={1};
    short tmp[FRAME_SIZE];
    float vad=0;
    float vad_prob;
    float E=0;
    if (count==50000000) break;
    if (++gain_change_count > 2821) {
      speech_gain = pow(10., (-40+(rand()%60))/20.);
      noise_gain = pow(10., (-30+(rand()%50))/20.);
      if (rand()%10==0) noise_gain = 0;
      noise_gain *= speech_gain;
      if (rand()%10==0) speech_gain = 0;
      gain_change_count = 0;
      rand_resp(a_noise, b_noise);
      rand_resp(a_sig, b_sig);
      lowpass = FREQ_SIZE * 3000./24000. * pow(50., rand()/(double)RAND_MAX);
      for (i=0;i<NB_BANDS;i++) {
        if (eband5ms[i]<<FRAME_SIZE_SHIFT > lowpass) {
          band_lp = i;
          break;
        }
      }
    }
    if (speech_gain != 0) {
      fread(tmp, sizeof(short), FRAME_SIZE, f1);
      if (feof(f1)) {
        rewind(f1);
        fread(tmp, sizeof(short), FRAME_SIZE, f1);
      }
      for (i=0;i<FRAME_SIZE;i++) x[i] = speech_gain*tmp[i];
      for (i=0;i<FRAME_SIZE;i++) E += tmp[i]*(float)tmp[i];
    } else {
      for (i=0;i<FRAME_SIZE;i++) x[i] = 0;
      E = 0;
    }
    if (noise_gain!=0) {
      fread(tmp, sizeof(short), FRAME_SIZE, f2);
      if (feof(f2)) {
        rewind(f2);
        fread(tmp, sizeof(short), FRAME_SIZE, f2);
      }
      for (i=0;i<FRAME_SIZE;i++) n[i] = noise_gain*tmp[i];
    } else {
      for (i=0;i<FRAME_SIZE;i++) n[i] = 0;
    }
    biquad(x, mem_hp_x, x, b_hp, a_hp, FRAME_SIZE);
    biquad(x, mem_resp_x, x, b_sig, a_sig, FRAME_SIZE);
    biquad(n, mem_hp_n, n, b_hp, a_hp, FRAME_SIZE);
    biquad(n, mem_resp_n, n, b_noise, a_noise, FRAME_SIZE);
    for (i=0;i<FRAME_SIZE;i++) xn[i] = x[i] + n[i];
    if (E > 1e9f) {
      vad_cnt=0;
    } else if (E > 1e8f) {
      vad_cnt -= 5;
    } else if (E > 1e7f) {
      vad_cnt++;
    } else {
      vad_cnt+=2;
    }
    if (vad_cnt < 0) vad_cnt = 0;
    if (vad_cnt > 15) vad_cnt = 15;

    if (vad_cnt >= 10) vad = 0;
    else if (vad_cnt > 0) vad = 0.5f;
    else vad = 1.f;

    frame_analysis(st, Y, Ey, x);
    frame_analysis(noise_state, N, En, n);
    for (i=0;i<NB_BANDS;i++) Ln[i] = log10(1e-2+En[i]);
    int silence = compute_frame_features(noisy, X, P, Ex, Ep, Exp, features, xn);
    pitch_filter(X, P, Ex, Ep, Exp, g);
    //printf("%f %d\n", noisy->last_gain, noisy->last_period);
    for (i=0;i<NB_BANDS;i++) {
      g[i] = sqrt((Ey[i]+1e-3)/(Ex[i]+1e-3));
      if (g[i] > 1) g[i] = 1;
      if (silence || i > band_lp) g[i] = -1;
      if (Ey[i] < 5e-2 && Ex[i] < 5e-2) g[i] = -1;
      if (vad==0 && noise_gain==0) g[i] = -1;
    }
    count++;
#if 0
    for (i=0;i<NB_FEATURES;i++) printf("%f ", features[i]);
    for (i=0;i<NB_BANDS;i++) printf("%f ", g[i]);
    for (i=0;i<NB_BANDS;i++) printf("%f ", Ln[i]);
    printf("%f\n", vad);
#endif
#if 1
    fwrite(features, sizeof(float), NB_FEATURES, stdout);
    fwrite(g, sizeof(float), NB_BANDS, stdout);
    fwrite(Ln, sizeof(float), NB_BANDS, stdout);
    fwrite(&vad, sizeof(float), 1, stdout);
#endif
#if 0
    compute_rnn(&noisy->rnn, g, &vad_prob, features);
    interp_band_gain(gf, g);
#if 1
    for (i=0;i<FREQ_SIZE;i++) {
      X[i].r *= gf[i];
      X[i].i *= gf[i];
    }
#endif
    frame_synthesis(noisy, xn, X);

    for (i=0;i<FRAME_SIZE;i++) tmp[i] = xn[i];
    fwrite(tmp, sizeof(short), FRAME_SIZE, fout);
#endif
  }
  fprintf(stderr, "matrix size: %d x %d\n", count, NB_FEATURES + 2*NB_BANDS + 1);
  fclose(f1);
  fclose(f2);
  fclose(fout);
  return 0;
}

#endif
