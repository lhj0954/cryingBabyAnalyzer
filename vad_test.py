import torch
import torchaudio
import wave
import struct
import math
import os

# 콘솔 출력 구분선
DIVIDER = "-" * 50

print("\n" + DIVIDER)
print(" [Silero VAD 모델 구동 테스트 (호환성 에러 해결판)]")
print(DIVIDER)

# 1. 모델 불러오기
print("[Step 1] 모델 로딩 중...")
try:
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  trust_repo=True)
    # save_audio는 에러가 나서 뺍니다. 우리가 직접 저장할 겁니다.
    (get_speech_timestamps, _, read_audio, VADIterator, collect_chunks) = utils
    print("   -> 모델 로드 완료 (Success)")
except Exception as e:
    print(f"   -> [Error] 모델 로드 실패: {e}")
    exit()

# 2. '가짜 사람 목소리' 파일 생성
filename = "my_test_audio.wav"
print(f"\n[Step 2] 테스트용 오디오 생성 (사람 목소리 주파수 모방)")
try:
    with wave.open(filename, "w") as f:
        f.setnchannels(1)      
        f.setsampwidth(2)      
        f.setframerate(16000)  
        
        audio_data = b''
        # 3초 길이 생성
        for i in range(16000 * 3):
            t = i / 16000
            # 1초 ~ 2.5초 사이에 소리 넣기
            if 1.0 <= t <= 2.5:
                # 150Hz 기본음 + 배음 합성
                val = math.sin(2 * math.pi * 150 * t)       
                val += 0.5 * math.sin(2 * math.pi * 300 * t) 
                val += 0.25 * math.sin(2 * math.pi * 450 * t) 
                sample = int(10000 * val)
            else:
                sample = 0
            sample = max(-32767, min(32767, sample))
            audio_data += struct.pack('<h', sample)
            
        f.writeframes(audio_data)
    print("   -> 오디오 파일 생성 완료 (Success)")
except Exception as e:
    print(f"   -> [Error] 파일 생성 실패: {e}")
    exit()

# 3. 파일 읽기
print(f"\n[Step 3] 오디오 파일 로드")
wav, sr = torchaudio.load(filename)

# 4. 모델 실행
print(f"\n[Step 4] VAD 추론 실행 중...")
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sr, threshold=0.3)

# 5. 결과 리포트 (증거 출력)
print("\n" + DIVIDER)
print(" [최종 결과 리포트]")

if len(speech_timestamps) > 0:
    print(f"   [감지 성공] 총 {len(speech_timestamps)}개의 구간을 찾았습니다.")
    
    # 증거 1: 정확한 시간 좌표
    for i, ts in enumerate(speech_timestamps):
        start_sec = ts['start'] / sr
        end_sec = ts['end'] / sr
        print(f"      - 구간 {i+1}: {start_sec:.3f}초 ~ {end_sec:.3f}초")

    # 증거 2: 시각화
    print("\n   [시각화 확인]")
    duration = 3.0 
    steps = 40     
    timeline_str = ""
    for i in range(steps):
        current_t = (i / steps) * duration
        is_speech = False
        for ts in speech_timestamps:
            if (ts['start']/sr) <= current_t <= (ts['end']/sr):
                is_speech = True
                break
        timeline_str += "■" if is_speech else "─"
            
    print(f"   0초 {timeline_str} 3초")
    print("       (─: 무음 / ■: 목소리 감지됨)")

    # 증거 3: 결과 파일 저장 (wave 모듈로 직접 저장)
    save_filename = "only_speech.wav"
    try:
        # 감지된 구간만 합친 텐서 가져오기
        cut_wav_tensor = collect_chunks(speech_timestamps, wav)
        # 텐서를 리스트로 변환
        cut_wav_list = cut_wav_tensor.tolist()
        
        with wave.open(save_filename, "w") as f:
            f.setnchannels(1)      
            f.setsampwidth(2)      
            f.setframerate(16000)  
            
            # Float -> Int16 변환 및 저장
            out_bytes = b''
            for sample in cut_wav_list:
                int_sample = int(sample * 32767)
                int_sample = max(-32768, min(32767, int_sample))
                out_bytes += struct.pack('<h', int_sample)
                
            f.writeframes(out_bytes)
            
        print(f"\n   [저장 완료] '{save_filename}' 파일이 정상적으로 저장되었습니다.")
        
    except Exception as e:
        print(f"\n   [저장 실패] 파일 저장 중 오류 발생: {e}")

else:
    print("   [결과] 감지된 구간이 없습니다.")

print(DIVIDER + "\n")
