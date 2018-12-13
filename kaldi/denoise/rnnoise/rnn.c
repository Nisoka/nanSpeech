/* Copyright (c) 2008-2011 Octasic Inc.
                 2012-2017 Jean-Marc Valin */
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

#include <stdint.h>
#include <math.h>
#include "opus_types.h"
#include "common.h"
#include "arch.h"
#include "rnn.h"
#include "rnn_data.h"
#include <stdio.h>

static OPUS_INLINE float fastPow2(float p) {
    float clipp = (p < -126) ? -126.0f : p;
    union {
        uint32_t i;
        float f;
    } v = {(uint32_t) ((1 << 23) * (clipp + 126.94269504f))};
    return v.f;
}

static OPUS_INLINE float
fastExp(float p) {
    return fastPow2(1.442695040f * p);
}

static OPUS_INLINE float tansig_approx(float x) {
    const float z = fastExp((WEIGHTS_SCALE * 2) * x);
    return (z - 1) / (z + 1);
}

static OPUS_INLINE float sigmoid_approx(float x) {
    return 1.f / (1.f + fastExp(-WEIGHTS_SCALE * x));
}

static OPUS_INLINE float relu(float x) {
    return WEIGHTS_SCALE * x * (x > 0);
}

void compute_dense(const DenseLayer *layer, float *output, const float *input) {
    int i, j;
    int M = layer->nb_inputs;
    int N = layer->nb_neurons;
    int stride = N;
    int idx = 0;
    if (layer->activation == ACTIVATION_SIGMOID) {
        for (i = 0; i < N; i++) { /* Compute update gate. */
            float sum = layer->bias[i];
            for (j = 0; j < M; j++) {
                idx = j * stride + i;
                sum += layer->input_weights[idx] * input[j];
            }
            output[i] = sigmoid_approx(sum);
        }
    } else if (layer->activation == ACTIVATION_TANH) {
        for (i = 0; i < N; i++) {    /* Compute update gate. */
            float sum = layer->bias[i];
            for (j = 0; j < M; j++) {
                idx = j * stride + i;
                sum += layer->input_weights[idx] * input[j];
            }
            output[i] = tansig_approx(sum);

        }

    } else if (layer->activation == ACTIVATION_RELU) {
        for (i = 0; i < N; i++) { /* Compute update gate. */
            float sum = layer->bias[i];
            for (j = 0; j < M; j++) {
                idx = j * stride + i;
                sum += layer->input_weights[idx] * input[j];
            }
            output[i] = relu(sum);
        }

    } else {
        *(int *) 0 = 0;
    }
}

void compute_gru(float *update, float *reset, const GRULayer *gru, float *state, const float *input) {
    int i, j;
    if ((reset == NULL) || (update == NULL)) {
        printf("[%s %d] malloc failed\n", __FUNCTION__, __LINE__);
        rnnoise_free(update);
        rnnoise_free(reset);
        return;
    }
    int M = gru->nb_inputs;
    int N = gru->nb_neurons;
    int stride = 3 * N;
    int idx = 0;
    if (M > N) {
        for (i = 0; i < N; i++) {
            /* Compute update gate. */
            float sum_update = gru->bias[i];
            /* Compute reset gate. */
            float sum_reset = gru->bias[N + i];
            for (j = 0; j < N; j++) {
                idx = j * stride + i;
                sum_update += gru->input_weights[idx] * input[j];
                sum_reset += gru->input_weights[N + idx] * input[j];
                sum_reset += gru->recurrent_weights[N + idx] * state[j];
                sum_update += gru->recurrent_weights[idx] * state[j];
            }
            for (j = N; j < M; j++) {
                idx = j * stride + i;
                sum_update += gru->input_weights[idx] * input[j];
                sum_reset += gru->input_weights[N + idx] * input[j];
            }
            update[i] = sigmoid_approx(sum_update);
            reset[i] = sigmoid_approx(sum_reset);
        }
        if (gru->activation == ACTIVATION_SIGMOID) {
            for (i = 0; i < N; i++) {
                /* Compute output. */
                float sum_output = gru->bias[(N << 1) + i];
                for (j = 0; j < N; j++) {
                    idx = j * stride + i;
                    sum_output += gru->recurrent_weights[(N << 1) + idx] * state[j] * reset[j];
                    sum_output += gru->input_weights[(N << 1) + idx] * input[j];
                }
                for (j = N; j < M; j++) {
                    idx = j * stride + i;
                    sum_output += gru->input_weights[(N << 1) + idx] * input[j];
                }
                state[i] = update[i] * state[i] + (1 - update[i]) * sigmoid_approx(sum_output);
            }
        } else if (gru->activation == ACTIVATION_TANH) {
            for (i = 0; i < N; i++) {
                /* Compute output. */
                float sum_output = gru->bias[(N << 1) + i];
                for (j = 0; j < N; j++) {
                    idx = j * stride + i;
                    sum_output += gru->recurrent_weights[(N << 1) + idx] * state[j] * reset[j];
                    sum_output += gru->input_weights[(N << 1) + idx] * input[j];
                }
                for (j = N; j < M; j++) {
                    idx = j * stride + i;
                    sum_output += gru->input_weights[(N << 1) + idx] * input[j];
                }
                state[i] = update[i] * state[i] + (1 - update[i]) * tansig_approx(sum_output);
            }
        } else if (gru->activation == ACTIVATION_RELU) {
            for (i = 0; i < N; i++) {
                /* Compute output. */
                float sum_output = gru->bias[(N << 1) + i];
                for (j = 0; j < N; j++) {
                    idx = j * stride + i;
                    sum_output += gru->recurrent_weights[(N << 1) + idx] * state[j] * reset[j];
                    sum_output += gru->input_weights[(N << 1) + idx] * input[j];
                }
                for (j = N; j < M; j++) {
                    idx = j * stride + i;
                    sum_output += gru->input_weights[(N << 1) + idx] * input[j];
                }
                state[i] = update[i] * state[i] + (1 - update[i]) * relu(sum_output);
            }
        } else {
            for (i = 0; i < N; i++) {
                /* Compute output. */
                float sum_output = gru->bias[(N << 1) + i];
                for (j = 0; j < N; j++) {
                    idx = j * stride + i;
                    sum_output += gru->recurrent_weights[(N << 1) + idx] * state[j] * reset[j];
                    sum_output += gru->input_weights[(N << 1) + idx] * input[j];
                }
                for (j = N; j < M; j++) {
                    idx = j * stride + i;
                    sum_output += gru->input_weights[(N << 1) + idx] * input[j];
                }
                state[i] = update[i] * state[i] + (1 - update[i]) * sum_output;
            }
        }
    } else {
        for (i = 0; i < N; i++) {
            /* Compute update gate. */
            float sum_update = gru->bias[i];
            /* Compute reset gate. */
            float sum_reset = gru->bias[N + i];
            for (j = M; j < N; j++) {
                idx = j * stride + i;
                sum_update += gru->input_weights[idx] * input[j];
                sum_reset += gru->input_weights[N + idx] * input[j];
                sum_reset += gru->recurrent_weights[N + idx] * state[j];
                sum_update += gru->recurrent_weights[idx] * state[j];
            }
            for (j = 0; j < M; j++) {
                idx = j * stride + i;
                sum_update += gru->input_weights[idx] * input[j];
                sum_reset += gru->input_weights[N + idx] * input[j];
            }
            update[i] = sigmoid_approx(sum_update);
            reset[i] = sigmoid_approx(sum_reset);
        }
        if (gru->activation == ACTIVATION_SIGMOID) {
            for (i = 0; i < N; i++) {
                /* Compute output. */
                float sum_output = gru->bias[(N << 1) + i];
                for (j = M; j < N; j++) {
                    idx = j * stride + i;
                    sum_output += gru->recurrent_weights[(N << 1) + idx] * state[j] * reset[j];
                    sum_output += gru->input_weights[(N << 1) + idx] * input[j];
                }
                for (j = 0; j < M; j++) {
                    sum_output += gru->input_weights[(N << 1) + idx] * input[j];
                }
                state[i] = update[i] * state[i] + (1 - update[i]) * sigmoid_approx(sum_output);
            }
        } else if (gru->activation == ACTIVATION_TANH) {
            for (i = 0; i < N; i++) {
                /* Compute output. */
                float sum_output = gru->bias[(N << 1) + i];
                for (j = M; j < N; j++) {
                    idx = j * stride + i;
                    sum_output += gru->recurrent_weights[(N << 1) + idx] * state[j] * reset[j];
                    sum_output += gru->input_weights[(N << 1) + idx] * input[j];
                }
                for (j = 0; j < M; j++) {
                    idx = j * stride + i;
                    sum_output += gru->input_weights[(N << 1) + idx] * input[j];
                }
                state[i] = update[i] * state[i] + (1 - update[i]) * tansig_approx(sum_output);
            }
        } else if (gru->activation == ACTIVATION_RELU) {
            for (i = 0; i < N; i++) {
                /* Compute output. */
                float sum_output = gru->bias[(N << 1) + i];
                for (j = M; j < N; j++) {
                    idx = j * stride + i;
                    sum_output += gru->recurrent_weights[(N << 1) + idx] * state[j] * reset[j];
                    sum_output += gru->input_weights[(N << 1) + idx] * input[j];
                }
                for (j = 0; j < M; j++) {
                    idx = j * stride + i;
                    sum_output += gru->input_weights[(N << 1) + idx] * input[j];
                }
                state[i] = update[i] * state[i] + (1 - update[i]) * relu(sum_output);
            }
        } else {
            for (i = 0; i < N; i++) {
                /* Compute output. */
                float sum_output = gru->bias[(N << 1) + i];
                for (j = M; j < N; j++) {
                    idx = j * stride + i;
                    sum_output += gru->recurrent_weights[(N << 1) + idx] * state[j] * reset[j];
                    sum_output += gru->input_weights[(N << 1) + idx] * input[j];
                }
                for (j = 0; j < M; j++) {
                    idx = j * stride + i;
                    sum_output += gru->input_weights[(N << 1) + idx] * input[j];
                }
                state[i] = update[i] * state[i] + (1 - update[i]) * sum_output;
            }
        }
    }
}

#define INPUT_SIZE 42

void compute_rnn(RNNState *rnn, float *gains, float *vad, const float *input) {
    float *cache = rnnoise_alloc(MAX_NEURONS * 5 * sizeof(float));
    if (cache == NULL) {
        printf("[%s %d] malloc failed\n", __FUNCTION__, __LINE__);
        return;
    }
    float *update = cache + (MAX_NEURONS * 3);
    float *reset = update + MAX_NEURONS;
    float *dense_out = cache;
    float *noise_input = cache;
    float *denoise_input = cache;
    compute_dense(&input_dense, dense_out, input);
    compute_gru(update, reset, &vad_gru, rnn->vad_gru_state, dense_out);
    compute_dense(&vad_output, vad, rnn->vad_gru_state);
    memcpy(noise_input + INPUT_DENSE_SIZE, rnn->vad_gru_state, VAD_GRU_SIZE * sizeof(float));
    memcpy(noise_input + INPUT_DENSE_SIZE + VAD_GRU_SIZE, input, INPUT_SIZE * sizeof(float));
    compute_gru(update, reset, &noise_gru, rnn->noise_gru_state, noise_input);
    memcpy(denoise_input, rnn->vad_gru_state, VAD_GRU_SIZE * sizeof(float));
    memcpy(denoise_input + VAD_GRU_SIZE, rnn->noise_gru_state, NOISE_GRU_SIZE * sizeof(float));
    memcpy(denoise_input + VAD_GRU_SIZE + NOISE_GRU_SIZE, input, INPUT_SIZE * sizeof(float));
    compute_gru(update, reset, &denoise_gru, rnn->denoise_gru_state, denoise_input);
    compute_dense(&denoise_output, gains, rnn->denoise_gru_state);
    rnnoise_free(cache);
}
