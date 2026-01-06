import torch
import torchaudio
import wave
import struct
import math
import os

# 콘솔 출력 깔끔하게 하기 위한 구분선
DIVIDER = "-" * 50

print("\n" + DIVIDER)
print("🛠️  Silero VAD 모델 구동 테스트 시작")
print(DIVIDER)

# 1. 모델 불러오기
print("[Step 1] 모델 로딩 중...")
try:
    model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  trust_repo=True)
    (get_speech_timestamps, _, _, _, _) = utils
    print("   ✅ 모델 로드 완료 (Success)")
except Exception as e:
    print(f"   ❌ 모델 로드 실패: {e}")
    exit()

# 2. 오디오 파일 생성
filename = "my_test_audio.wav"
print(f"\n[Step 2] 테스트용 오디오 생성 (파일명: {filename})")
try:
    with wave.open(filename, "w") as f:
        f.setnchannels(1)      
        f.setsampwidth(2)      
        f.setframerate(16000)  
        
        audio_data = b''
        # 3초 길이, 1~2초 구간에 400Hz Sine Wave 생성
        for i in range(16000 * 3):
            t = i / 16000
            if 1.0 <= t <= 2.0:
                sample = int(20000 * math.sin(2 * math.pi * 400 * t))
            else:
                sample = 0
            audio_data += struct.pack('<h', sample)
            
        f.writeframes(audio_data)
    print("   ✅ 오디오 파일 생성 완료 (Success)")
except Exception as e:
    print(f"   ❌ 파일 생성 실패: {e}")
    exit()

# 3. 파일 읽기 및 전처리
print(f"\n[Step 3] 오디오 파일 로드 및 전처리")
try:
    wav, sr = torchaudio.load(filename)
    print(f"   ℹ️  Sample Rate: {sr}Hz / Shape: {wav.shape}")
except Exception as e:
    print(f"   ❌ 파일 읽기 실패: {e}")
    exit()

# 4. 모델 실행
print(f"\n[Step 4] VAD 추론(Inference) 실행 중...")
try:
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sr)
    print("   ✅ 추론 완료 (Success)")
except Exception as e:
    print(f"   ❌ 추론 중 에러 발생: {e}")
    exit()

# 5. 최종 결과 리포트
print("\n" + DIVIDER)
print("📊  [테스트 결과 리포트]")

if len(speech_timestamps) > 0:
    print(f"   📍 감지된 음성 구간: {len(speech_timestamps)}개")
    for i, ts in enumerate(speech_timestamps):
        start_sec = ts['start'] / sr
        end_sec = ts['end'] / sr
        print(f"      - 구간 {i+1}: {start_sec:.3f}초 ~ {end_sec:.3f}초")
else:
    print("   📍 감지된 음성 구간: 0개 (Not Detected)")

print("\n🚀 [시스템 상태 확인]")
print("   - 라이브러리 호환성: 정상")
print("   - 모델 로드 및 실행: 정상")
print("   - 결론: Silero VAD 사용 준비 완료")
print(DIVIDER + "\n")