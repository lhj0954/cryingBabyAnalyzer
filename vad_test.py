import torch
import torchaudio
import wave
import struct
import math
import os

# 콘솔 출력 구분선
DIVIDER = "-" * 50

print("\n" + DIVIDER)
print("🛠️  Silero VAD 모델 구동 테스트 (음성 감지 시연용)")
print(DIVIDER)

# 1. 모델 불러오기
print("[Step 1] 모델 로딩 중...")
try:
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  trust_repo=True)
    (get_speech_timestamps, _, _, _, _) = utils
    print("   ✅ 모델 로드 완료")
except Exception as e:
    print(f"   ❌ 모델 로드 실패: {e}")
    exit()

# 2. '가짜 사람 목소리' 파일 생성
# 단순 삐~ 소리 대신, 사람 목소리 톤(기본음+배음)을 흉내 낸 소리를 만듭니다.
filename = "my_test_audio.wav"
print(f"\n[Step 2] 테스트용 오디오 생성 (사람 목소리 흉내)")
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
                # 150Hz(남자 저음) + 배음들을 섞어서 목소리처럼 들리게 함
                val = math.sin(2 * math.pi * 150 * t)       # 기본음
                val += 0.5 * math.sin(2 * math.pi * 300 * t) # 배음 1
                val += 0.25 * math.sin(2 * math.pi * 450 * t) # 배음 2
                sample = int(10000 * val)
            else:
                sample = 0
            # 범위를 벗어나지 않게 클리핑
            sample = max(-32767, min(32767, sample))
            audio_data += struct.pack('<h', sample)
            
        f.writeframes(audio_data)
    print("   ✅ 오디오 파일 생성 완료")
except Exception as e:
    print(f"   ❌ 파일 생성 실패: {e}")
    exit()

# 3. 파일 읽기
print(f"\n[Step 3] 오디오 파일 로드")
wav, sr = torchaudio.load(filename)

# 4. 모델 실행 (강제 감지 모드)
print(f"\n[Step 4] VAD 추론 실행 중...")
# threshold=0.3 : 감지 기준을 약간 낮춰서 기계음도 잘 잡게 설정
speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sr, threshold=0.3)

# 5. 결과 리포트
print("\n" + DIVIDER)
print("📊  [최종 결과 리포트]")

if len(speech_timestamps) > 0:
    print(f"   🎉 감지 성공! 총 {len(speech_timestamps)}개의 구간을 찾았습니다.")
    for i, ts in enumerate(speech_timestamps):
        start_sec = ts['start'] / sr
        end_sec = ts['end'] / sr
        print(f"      👉 구간 {i+1}: {start_sec:.3f}초 ~ {end_sec:.3f}초 (음성 인식됨)")
else:
    print("   ⚠️ 여전히 감지되지 않음 (볼륨이나 주파수 조정 필요)")

print(DIVIDER + "\n")
