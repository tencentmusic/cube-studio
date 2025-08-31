from openai import OpenAI
# 新版本openai
client = OpenAI(api_key='xx',base_url='http://192.168.3.7:20170/v1')

def get_completion(prompt, model="chatglm2-6b",stream=True):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
        stream=stream
    )
    if stream:
        message = ''
        for chunk in response:
            try:
                message_one = chunk.choices[0].delta.content
                if message_one:
                    message+=message_one
                    print(message_one, end="", flush=True)
            except Exception as e:
                pass

        return message
    else:
        return response.choices[0].message.content

prompt = "你好"

print(get_completion(prompt))
