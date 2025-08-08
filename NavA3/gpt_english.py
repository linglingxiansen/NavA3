key=''


from openai import OpenAI

import base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

client = OpenAI(
        base_url="https://jeniya.top/v1",
        api_key=key
    )
def gpt_object(prompt):
    client = OpenAI(
        base_url="https://jeniya.top/v1",
        api_key=key
    )

    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": f"Based on this answer and thinking process: {prompt}, tell me what object I should look for and directly return the name of the object to me."},

    ]
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def gpt_room(prompt):
    client = OpenAI(
        base_url="https://jeniya.top/v1",
        api_key=key
    )

    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": f"Based on this answer and thinking process: {prompt}, tell me what room I should look for and directly return the name of the room to me."},

    ]
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def gpt_thinking(prompt):
    client = OpenAI(
        base_url="https://jeniya.top/v1",
        api_key=key
    )
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": f"You need to find this object: {prompt}. Now given this top-down scene view and several optional regions, please think about what object you should find to complete the instruction and where you should look for this object. Please show your thinking process and give your answer at the end."},
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(f'globalmap.png')}",
            },
            },
        ],
        }
    ],
    max_tokens=300,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content




def gpt_image(object_name,i):
    client = OpenAI(
        base_url="https://jeniya.top/v1",
        api_key=key
    )
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": f"这张图片中有{object_name}吗？只需要返回yes或者no,不需要其他信息"},
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(f'rgb{i}.png')}",
            },
            },
        ],
        }
    ],
    max_tokens=300,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content
# 


def yuyinzhuanwenzi():
    audio_file = open("/home/lingfeng/zed/nav/my_recording.wav", "rb")
    transcription = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
    )
    print(transcription.text)
    return transcription.text

from playsound import playsound

def bofang(prompt):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=prompt,
    )

    response.stream_to_file("output.mp3")
    playsound('output.mp3')

# prompt = ' I want  to drink coffee'

# gpt_room(prompt)



def gpt_point(prompt):
    client = OpenAI(
        base_url="https://jeniya.top/v1",
        api_key=key
    )
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Locate several points on the {prompt}. Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], where each tuple contains the x and y coordinates of a point satisfying the conditions above. The coordinates should be rounded to two decimal places, indicating the absolute pixel locations of the points in the image. Please directly return the points: [(x1, y1), (x2, y2), ...]."},
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(f'rgb.png')}",
            },
            },
        ],
        }
    ],
    max_tokens=300,
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content





