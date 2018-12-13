#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "util/common-utils.h"
#include "feat/wave-reader.h"
#include "feat/signal.h"

#ifdef __cplusplus
extern "C"
{
#endif
#include "rnnoise.h"
#ifdef __cplusplus
}
#endif

  
using namespace kaldi;
//ref :https://github.com/cpuimage/resampler
//poly s16 version
void poly_resample_s16(const int16_t *input, int16_t *output, int in_frames, int out_frames, int channels) {
    double scale = (1.0 * in_frames) / out_frames;
    int head = (int) (1.0 / scale);
    float pos = 0;
    for (int i = 0; i < head; i++) {
        for (int c = 0; c < channels; c++) {
            int sample_1 = input[0 + c];
            int sample_2 = input[channels + c];
            int sample_3 = input[(channels << 1) + c];
            int poly_3 = sample_1 + sample_3 - (sample_2 << 1);
            int poly_2 = (sample_2 << 2) + sample_1 - (sample_1 << 2) - sample_3;
            int poly_1 = sample_1;
            output[i * channels + c] = (int16_t) ((poly_3 * pos * pos + poly_2 * pos) * 0.5f + poly_1);
        }
        pos += scale;
    }
    double in_pos = head * scale;
    for (int n = head; n < out_frames; n++) {
        int npos = (int) in_pos;
        pos = in_pos - npos + 1;
        for (int c = 0; c < channels; c++) {
            int sample_1 = input[(npos - 1) * channels + c];
            int sample_2 = input[(npos + 0) * channels + c];
            int sample_3 = input[(npos + 1) * channels + c];
            int poly_3 = sample_1 + sample_3 - (sample_2 << 1);
            int poly_2 = (sample_2 << 2) + sample_1 - (sample_1 << 2) - sample_3;
            int poly_1 = sample_1;
            output[n * channels + c] = (int16_t) ((poly_3 * pos * pos + poly_2 * pos) * 0.5f + poly_1);
        }
        in_pos += scale;
    }
}

void denoise_proc(int16_t *buffer, uint32_t buffen_len) {
#define  FRAME_SIZE   480
    DenoiseState *st;
    st = rnnoise_create();
    int16_t patch_buffer[FRAME_SIZE];
    if (st != NULL) {
        uint32_t frames = buffen_len / FRAME_SIZE;
        uint32_t lastFrame = buffen_len % FRAME_SIZE;
        for (int i = 0; i < frames; ++i) {
            rnnoise_process_frame(st, buffer, buffer);
            buffer += FRAME_SIZE;
        }
        if (lastFrame != 0) {
            memset(patch_buffer, 0, FRAME_SIZE * sizeof(int16_t));
            memcpy(patch_buffer, buffer, lastFrame * sizeof(int16_t));
            rnnoise_process_frame(st, patch_buffer, patch_buffer);
            memcpy(buffer, patch_buffer, lastFrame * sizeof(int16_t));
        }
    }
    rnnoise_destroy(st);
}

void rnnDeNoise(char *input_wave_file, char *output_wave_file) {
    uint32_t sampleRateLimit = 48000;
    int channelLimit = 1;
    
    // kaldi classType
    WaveData input_wave;
    {
      WaveHolder waveholder;
      Input ki(input_wave_file);
      waveholder.Read(ki.Stream());
      input_wave = waveholder.Value();
    }

    const Matrix<BaseFloat> &input_matrix = input_wave.Data();
    BaseFloat samp_freq_input = input_wave.SampFreq();
    int32 num_samp_input = input_matrix.NumCols(),  // #samples in the input
          num_input_channel = input_matrix.NumRows();  // #channels in the input

    KALDI_LOG << "sampling frequency of input: " << samp_freq_input
                  << " #samples: " << num_samp_input
                  << " #channel: " << num_input_channel;

    int16_t *data_in = (int16_t *) malloc(num_samp_input * sizeof(int16_t));

    for(int i = 0; i < num_samp_input; i++){
      data_in[i] = input_matrix(0, i);
    }

    uint32_t num_samp_output = (uint32_t) (num_samp_input * ((float) sampleRateLimit / samp_freq_input));
    int16_t *data_out = (int16_t *) malloc(num_samp_output * sizeof(int16_t));

    if (data_in != NULL && data_out != NULL){
      // resample the input wav sample-rate to 48K
      poly_resample_s16(data_in, data_out, num_samp_input, num_samp_output, channelLimit);
      denoise_proc(data_out, num_samp_output);
      // resample the output wav sample-rate to input sample-rate
      poly_resample_s16(data_out, data_in, num_samp_output, num_samp_input, channelLimit);

      
      // the rnnoise limit the channel must be 1, so all the channel input will get the channel 1
      Matrix<BaseFloat> out_matrix(channelLimit, num_samp_input);
      for(int i = 0; i < num_samp_input; i++){
        out_matrix(0, i) = data_in[i];
      }
      
      WaveData out_wave(samp_freq_input, out_matrix);
      Output ko(output_wave_file, false);
      out_wave.Write(ko.Stream());
      
      free(data_in);
      free(data_out);
    } else {
      
      if (data_in)
        free(data_in);
      if (data_out)
        free(data_out);
      
    }
}


int main(int argc, char **argv) {
    if (argc < 3){
      printf("usage: rnnoise inputfile outputfile");
      return -1;
    }

    char *in_file = argv[1];
    char *out_file = argv[2];
    rnnDeNoise(in_file, out_file);
    return 0;
}
