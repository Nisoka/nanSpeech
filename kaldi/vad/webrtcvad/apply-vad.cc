/* Created on 2017-03-01
 * Author: Binbin Zhang
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <vector>

#include "util/parse-options.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"
#include "feat/signal.h"

#include "vad.h"

using namespace kaldi;

int main(int argc, char *argv[]) {
    const char *usage = "Apply energy vad for input wav file\n"
                        "Usage: vad-test wav_in_file\n";
    ParseOptions po(usage);

    int channelLimit = 1;
    int frame_len = 10; // 10 ms
    int mode = 0; 
    po.Register("frame-len", &frame_len, "frame length in millionsenconds, must be 10/20/30");
    po.Register("mode", &mode, "vad mode");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
        po.PrintUsage();
        exit(1);
    }

    std::string input_wave_file = po.GetArg(1), 
        output_wave_file = po.GetArg(2);


    // kaldi classType
    WaveData input_wave;
    {
      WaveHolder waveholder;
      Input ki(input_wave_file);
      waveholder.Read(ki.Stream());
      input_wave = waveholder.Value();
    }

    const Matrix<BaseFloat> &input_matrix = input_wave.Data();
    BaseFloat sample_rate = input_wave.SampFreq();
    int32 num_sample = input_matrix.NumCols(),  // #samples in the input
          num_input_channel = input_matrix.NumRows();  // #channels in the input

    KALDI_LOG << "sampling frequency of input: " << sample_rate
              << " #samples: " << num_sample
              << " #channel: " << num_input_channel;

    // sample_rate is  num point of 1s
    // frame_len is n(ms), so the frame point is frame_len*sample_rate/1000
    int num_point_per_frame = (int)(frame_len * sample_rate / 1000);

    // int num_frames = num_sample / num_point_per_frame;

    int num_frames_speech = 0;


    // input data
    short *data = (short *)calloc(sizeof(short), num_sample);

    // limit only 0 channel
    for (int i = 0; i < num_sample; i++) {
      data[i] = input_matrix(0, i);
    }
   
    
    
    Vad vad(mode);    
    std::vector<bool> vad_reslut;
    for (int i = 0; i < num_sample; i += num_point_per_frame) {
        // last frame 
        if (i + num_point_per_frame > num_sample)
          break;
        
        bool tags = vad.IsSpeech(data+i, num_point_per_frame, sample_rate);
        
        vad_reslut.push_back(tags);
        
        if (tags)
          num_frames_speech++;
    }

    
    // speech points == frames * vad_true_frames
    int num_speech_sample = num_frames_speech * num_point_per_frame;
    short *speech_data = (short *)calloc(sizeof(short), num_speech_sample);
    
    int speech_cur = 0;
    for (int i = 0; i < vad_reslut.size(); i++) {
        // num point of one speech frame 
        if (vad_reslut[i]) {
          memcpy(speech_data + speech_cur * num_point_per_frame,
                 data + i * num_point_per_frame, 
                 num_point_per_frame * sizeof(short));
          speech_cur++;
        }
    }

    // the rnnoise limit the channel must be 1, so all the channel input will get the channel 1
    Matrix<BaseFloat> out_matrix(channelLimit, num_speech_sample);
    for(int i = 0; i < num_sample; i++){
      out_matrix(0, i) = speech_data[i];
    }

    WaveData out_wave(sample_rate, out_matrix);
    Output ko(output_wave_file, false);
    out_wave.Write(ko.Stream());

    
    free(data);
    free(speech_data);
    return 0;
}


