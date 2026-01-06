import torch
import torchaudio
import wave
import struct
import math
import os

# 콘솔 출력 구분선
DIVIDER = "-" * 50

print("\n" + DIVIDER)
print(" [Silero VAD 모델 구동 테스트 (소리 증폭/연장 버전)]")
print(DIVIDER)

# 1. 모델 불러오기
print("[Step 1] 모델 로딩 중...")
try:
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  trust_repo=True)
    (get_speech_timestamps, _, _, _, _) = utils
    print("   -> 모델 로드 완료")
except Exception as e:
    print(f"   -> [Error] 모델 로드 실패: {e}")
    exit()

# 2. 테스트용 오디오 파일 생성 (길게, 크게!)
filename = "my_test_audio.wav"
print(f"\n[Step 2] 테스트용 오디오 생성 (5초 길이)")
try:
    with wave.open(filename, "w") as f:
        f.setnchannels(1)      
        f.setsampwidth(2)      
        f.setframerate(16000)  
        
        audio_data = b''
        # [변경] 5초 길이로 늘림 (넉넉하게)
        for i in range(16000 * 5):
            t = i / 16000
            # [변경] 1.0초 ~ 4.0초 (총 3초간) 소리 냄
            if 1.0 <= t <= 4.0:
                val = math.sin(2 * math.pi * 150 * t)       
                val += 0.5 * math.sin(2 * math.pi * 300 * t) 
                val += 0.25 * math.sin(2 * math.pi * 450 * t) 
                # [변경] 볼륨을 2배 키움 (10000 -> 20000)
                sample = int(20000 * val)
            else:
                sample = 0
            sample = max(-32767, min(32767, sample))
            audio_data += struct.pack('<h', sample)
            
        f.writeframes(audio_data)
    print("   -> 오디오 파일 생성 완료")
except Exception as e:
    print(f"   -> [Error] 파일 생성 실패: {e}")
    exit()

# 3. 파일 읽기
print(f"\n[Step 3] 오디오 파일 로드")
wav, sr = torchaudio.load(filename)

# 4. 모델 실행
print(f"\n[Step 4] VAD 추론 실행 중...")
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sr, threshold=0.3)

# 5. 결과 리포트 및 수동 저장
print("\n" + DIVIDER)
print(" [최종 결과 리포트]")

if len(speech_timestamps) > 0:
    print(f"   [감지 성공] 총 {len(speech_timestamps)}개의 구간을 찾았습니다.")
    
    # 5-1. 구간 정보 출력
    for i, ts in enumerate(speech_timestamps):
        start_sec = ts['start'] / sr
        end_sec = ts['end'] / sr
        print(f"      - 구간 {i+1}: {start_sec:.3f}초 ~ {end_sec:.3f}초")

    # 5-2. 시각화
    print("\n   [시각화 확인]")
    duration = 5.0 # 전체 5초
    steps = 50     
    timeline_str = ""
    for i in range(steps):
        current_t = (i / steps) * duration
        is_speech = False
        for ts in speech_timestamps:
            if (ts['start']/sr) <= current_t <= (ts['end']/sr):
                is_speech = True
                break
        timeline_str += "■" if is_speech else "─"
    print(f"   0초 {timeline_str} 5초")
    print("       (─: 무음 / ■: 목소리 감지됨)")

    # 5-3. 수동 슬라이싱 및 저장
    save_filename = "only_speech.wav"
    try:
        combined_tensor = torch.Tensor()
        segments = []
        for ts in speech_timestamps:
            segment = wav[0][ts['start']:ts['end']]
            segments.append(segment)
            
        if segments:
            combined_tensor = torch.cat(segments)
        
        pcm_data = combined_tensor.tolist()
        
        print(f"\n   [데이터 확인] 저장할 샘플 개수: {len(pcm_data)}개")

        if len(pcm_data) > 0:
            with wave.open(save_filename, "w") as f:
                f.setnchannels(1)      
                f.setframerate(16000)
                f.setsampwidth(2)
                
                out_bytes = b''
                for sample in pcm_data:
                    int_val = int(max(-1.0, min(1.0, sample)) * 32767)
                    out_bytes += struct.pack('<h', int_val)
                
                f.writeframes(out_bytes)
                
            file_size = os.path.getsize(save_filename)
            print(f"   [저장 완료] '{save_filename}' (파일 크기: {file_size} bytes)")
        else:
            print("   [오류] 저장할 데이터가 비어있습니다.")
        
    except Exception as e:
        print(f"\n   [저장 실패] 파일 저장 중 오류 발생: {e}")

else:
    print("   [결과] 감지된 구간이 없습니다.")

print(DIVIDER + "\n")
