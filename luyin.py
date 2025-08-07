import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr 
import pyttsx3

from openai import OpenAI
import time
from gpt_english import yuyinzhuanwenzi,bofang


engine = pyttsx3.init()
def record_audio(filename="my_recording.wav", duration=5, sample_rate=44100):
    """
    使用麦克风录制音频并保存为wav文件
    :param filename: 保存的文件名
    :param duration: 录音时长，单位秒
    :param sample_rate: 采样率，默认为44100Hz
    """
    print("开始录音...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=2, dtype='int16')
    sd.wait()  # 等待录音完成
    print("录音完成！保存文件中...")

    write(filename, sample_rate, audio_data)
    print(f"音频已保存为 {filename}")

    instruction = yuyinzhuanwenzi()

    text = 'OK, I have received the instruction: '+instruction+' , let me think about it.'
    print(text)
    bofang(text)
    # time.sleep(5)

    # exit()
    return instruction


def bofang1(text):
    
    text += '。。。。。。'
    voice = engine.getProperty('voices')
    # print(voice[12].id)
    # engine.setProperty('voice','english')
    # engine.say('hello')
    # engine.runAndWait()
    # for i in voice:
        # print(i.id,i.name)
    # print(voice)
    # print(voice[12].id)
    engine.setProperty('voice','Chinese (Mandarin)')

    rate = engine.getProperty('rate')
    # print(rate)
    # exit()
    engine.setProperty('rate',rate)
    engine.setProperty('volume',0.5)

    engine.say(text)
    engine.runAndWait()
# 调用录音函数
def main():
    record_audio("my_recording.wav", duration=5)

    key='sk-MJZFQ5CjjHnYfkJ7hZrYQabiUkitcdZGkUicYCmIklpFNcBo'
    client = OpenAI(
        base_url="https://jeniya.top/v1",
        api_key=key
    )

    # 基础转录
    audio_file = open("/home/lingfeng/zed/nav/my_recording.wav", "rb")
    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
    )
    print(transcription.text)


    engine = pyttsx3.init()

    voice = engine.getProperty('voices')
    # print(voice[12].id)
    # engine.setProperty('voice','english')
    # engine.say('hello')
    # engine.runAndWait()
    # for i in voice:
        # print(i.id,i.name)
    # print(voice)
    # print(voice[12].id)
    engine.setProperty('voice','Chinese (Mandarin)')

    rate = engine.getProperty('rate')
    print(rate)
    engine.setProperty('rate',rate-60)
    engine.setProperty('volume',0.5)

    engine.say(transcription.text)
    engine.runAndWait()


