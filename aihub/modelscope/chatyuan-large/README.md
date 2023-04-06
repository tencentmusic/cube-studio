
ChatYuan: 元语功能型对话大模型

这个模型可以用于问答、结合上下文做对话、做各种生成任务，包括创意性写作，也能回答一些像法律、新冠等领域问题。它基于PromptCLUE-large结合数亿条功能对话多轮对话数据进一步训练得到。

<a href='https://www.cluebenchmarks.com/clueai.html'>PromptCLUE-large</a>在1000亿token中文语料上预训练，累计学习1.5万亿中文token，并且在数百种任务上进行Prompt任务式训练。针对理解类任务，如分类、情感分析、抽取等，可以自定义标签体系；针对多种生成任务，可以进行采样自由生成。 

<a href='https://www.clueai.cn/chat' target="__blank">在线Demo(微信搜索小程序“元语智能”)</a> &nbsp; | 
  <a href='https://www.clueai.cn' target="__blank">使用API(large版)</a> &nbsp; | 
 &nbsp; <a href='https://github.com/clue-ai/ChatYuan' target="__blank">Github项目地址</a>&nbsp; |
  &nbsp;<a href='https://colab.research.google.com/drive/1ZcLIJuemiojigrfjbsDMBWrX7JqXZX6I?usp=sharing' target="__blank">Colab在线试用</a> &nbsp; |
  &nbsp;<a href='https://mp.weixin.qq.com/s/-axa6XcjGl_Koeq_OrDq8w' target="__blank">文章介绍</a> 



## 期望模型使用方式及适用范围

### 如何使用

在安装完成ModelScope之后即可使用ChatYuan的能力，解决全中文NLP问题


#### 代码范例

加载模型：
```python
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.models.nlp import T5ForConditionalGeneration
from modelscope.preprocessors import TextGenerationT5Preprocessor
```

使用模型进行预测推理方法：
```python
model = T5ForConditionalGeneration.from_pretrained('ClueAI/ChatYuan-large', revision='v1.0.0')
preprocessor = TextGenerationT5Preprocessor(model.model_dir)
pipeline_t2t = pipeline(task=Tasks.text2text_generation, model=model, preprocessor=preprocessor)

def preprocess(text):
  text = text.replace("\n", "\\n").replace("\t", "\\t")
  return text

def postprocess(text):
  return text.replace("\\n", "\n").replace("\\t", "\t")

def answer(text, sample=True, top_p=1, temperature=0.7):
  '''sample：是否抽样。生成任务，可以设置为True;
  top_p：0-1之间，生成的内容越多样'''
  text = preprocess(text)
  
  if not sample:
    out_text = pipeline_t2t(text, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, num_beams=1, length_penalty=0.6)
  else:
    out_text = pipeline_t2t(text, return_dict_in_generate=True, output_scores=False, max_new_tokens=512, do_sample=True, top_p=top_p, temperature=temperature, no_repeat_ngram_size=3)
  
  return postprocess(out_text["text"])
  
```
## 问答、写作与功能型助手
```python

input_text0 = "帮我写一个请假条，我因为新冠不舒服，需要请假3天，请领导批准"
input_text1 = "你能干什么"
input_text2 = "用英文写一封道歉的邮件，表达因为物流延误，不能如期到达，我们可以赔偿贵公司所有损失"
input_text3 = "写一个文章，题目是未来城市"
input_text4 = "写一个诗歌，关于冬天"
input_text5 = "从南京到上海的路线"
input_text6 = "学前教育专业岗位实习中，在学生方面会存在问题，请提出改进措施。800字"
input_text7 = "根据标题生成文章：标题：屈臣氏里的化妆品到底怎么样？正文：化妆品，要讲究科学运用，合理搭配。屈臣氏起码是正品连锁店。请继续后面的文字。"
input_text8 = "帮我对比几款GPU，列出详细参数对比，并且给出最终结论"
input_list = [input_text0, input_text1, input_text2, input_text3, input_text4, input_text5, input_text6, input_text7, input_text8]
for i, input_text in enumerate(input_list):
  input_text = "用户：" + input_text + "\n小元："
  print(f"示例{i}".center(50, "="))
  output_text = answer(input_text)
  print(f"{input_text}{output_text}")

```

问答、生成与创意性写作：
 
    =======================示例1========================  
    用户：你能干什么  
    小元：您好!我是元语AI。我可以回答您的问题、写文章、写作业、翻译，对于一些法律等领域的问题我也可以给你提供信息。  
    =======================示例2========================  
    用户：写一封英文商务邮件给英国客户，表达因为物流延误，不能如期到达，我们可以赔偿贵公司所有损失  
    小元：
    Dear [Name],

    I'm sorry to inform you that we have been doing our business to you for your delay. We understand that we can compensate you for your losses due to the delay in our logistics.

    We understand that we can make sure we can compensate you for your losses due to the delay.

    Thank you for your understanding and we are very grateful for your efforts.

    Sincerely,

    [Your Name]
    =======================示例3========================  
    用户：写一个文章，题目是未来城市  
    小元：未来的城市是怎样的呢？  
      
    未来城市拥有很多不同的功能，如智慧城市、智能交通系统、远程医疗系统、智能家居系统、智慧医疗系统等。未来城市还使用物联网技术，可以远程控制，使用远程操控和传感器来监控城市的发展，帮助他们解决各种城市问题。  
      
    未来的城市还采用互联网技术，可以让城市变得更智能，让城市变得更智慧，让每一个人都能更轻松地生活。未来城市的智能设备也可以让城市的使用者更方便地掌握他们的生活，比如智能的智能交通系统和远程医疗系统和智能的安防系统等，可以提供他们更多的便利。  
    总之，未来城市的愿景是，让所有城市的使用者都能拥有一个美好的未来，让他们可以在自己的城市享受美好的生活。  
    =======================示例4========================  
    用户：写一个诗歌，关于冬天  
    小元：冬夜寂静冷， 
    云在天边飘，  
    冰封白雪上，  
    寒冷像一场雪。 
     
    雪花融化成冰，  
    像那雪花飘洒，  
    在寒冷的冬天，  
    感受春天的喜悦。  
     冬日里， 
    风雪渐消，  
    一片寂静，  
    把快乐和温暖带回家。    
    =======================示例5========================  
    用户：从南京到上海的路线  
    小元：你好，南京到上海的路线如下：  
      
    1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。  
     
    2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。  
      
    3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站。    
     
    =======================示例6========================  
    用户：学前教育专业岗位实习中，在学生方面会存在问题，请提出改进措施。800字  
    小元：在学前教育专业岗位实习中，在学生方面会存在一些问题，如学生缺少专业知识、缺乏合作意识、缺乏基本的责任心等等，需要我们采取措施，来解决这些问题。  
      
    首先，在实习过程中，我们需要培养学生的团队意识，加强学生与老师之间的合作，加强与家长之间的沟通，提高他们的沟通能力。另外，我们也需要加强学生的独立性，让学生学会独立思考，学会和老师、家长进行沟通，培养其独立性。  
    此外，在日常的工作中，我们也要为学生提供适当的机会和机会，让学生多接触社会、多参加社会活动，加深他们对社会、学校的认识，培养他们主动、积极和创新的精神。  
    最后，我们还需要重视学生在实习中出现的问题，尤其是学生缺乏合作，有依赖心理，不懂得分享，有拖延行为等，这些学生应该引起我们的重视，制定合理的改进措施，改善这些问题。   
    =======================示例7========================  
    用户：根据标题生成文章：屈臣氏里的化妆品到底怎么样？
    小元：近日，一条关于屈臣氏化妆品的微博引发网友热议。“在屈臣士商场里，有一些护肤品售价动辄上千元，部分化妆品甚至还卖到几折。屈臣仕化妆品专柜工作人员表示：“屈臣购商城的化妆品都是正规厂家生产，都是通过国家药品监督管理局的正规渠道进入市场的，并且都是正品。”“该微博发出后，引起网友们的热议，甚至不少网友开始搜索屈臣氏旗舰店来买产品。“屈氏化妆品真的值得买吗？“记者在屈士仕商场内看到，屈臣专卖店的货架上摆满了屈臣旗下的各种化妆品，有精华、面霜、眼霜等。““屈士士，这一款我买了，这款是主打美白的。”市民王女士说，她去年买了三瓶屈臣芮，其中一款是保湿面膜，还有一款是护手霜。“记者通过电话联系了屈臣臣芮的销售小姐，她说屈臣城里没有卖屈臣诗的化妆品，只有屈臣肌妍的化妆品。“
    =======================示例8========================  
    用户：帮我对比几款GPU，列出详细参数对比，并且给出最终结论  
    小元：一、显存容量：  
      
    1 GeForce GTX 880 显存类型：GDDR5  
    2 Geforce RTX 1060 显存在类型：PhysX物理加速 
    3 Ge force MX150 显存在于类型：SLI  
    4 GetoDirectX 11：支持DX11  
    二、显显存带宽：  
    1.GDDR4X 显显容量：256MB  
    2.GDDR6X 最大显存范围：8GB  
    3.GDDR3 显在带宽：120GB  
    4.GDDR7 显适用于2GB显存 
    三、显效时间：  
    1.4 GB/s 
    2. 5.5 ms 
    3. 5 ms     

## 多轮对话

```python
input_text = ["你好","新冠什么症状？","可以吃什么药？"]
answer_text = ["您好!我是元语AI。我可以回答您的问题、写文章、写作业、翻译，对于一些法律等领域的问题我也可以给你提供信息", "新冠是指新型冠状病毒，其症状包括发热、干咳、乏力、嗅味觉减退、呼吸困难等。", "根据您提供的病史，目前没有明确的抗新冠病毒的药物，建议您在家进行自我隔离，避免与他人接触，多喝开水，清淡易消化饮食，避免熬夜和过度劳累，适当进行户外活动。"]
context = "\n".join([f"用户：{input_text[i]}\n小元：{answer_text[i]}" for i in range(len(input_text))])
print(context)

input_text = "用什么后遗症么？"
print(f"示例".center(50, "="))
input_text = context + "\n用户：" + input_text + "\n小元："
output_text = answer(input_text)
print(f"{input_text}{output_text}")


``` 
========================示例========================  
用户：你好. 
小元：您好!我是元语AI。我可以回答您的问题、写文章、写作业、翻译，对于一些法律等领域的问题我也可以给你提供信息. 
用户：新冠什么症状？  
小元：新冠是指新型冠状病毒，其症状包括发热、干咳、乏力、嗅味觉减退、呼吸困难等。  
用户：可以吃什么药？  
小元：根据您提供的病史，目前没有明确的抗新冠病毒的药物，建议您在家进行自我隔离，避免与他人接触，多喝开水，清淡易消化饮食，避免熬夜和过度劳累，适当进行户外活动。  
用户：用什么后遗症么？  
小元：目前还没有人具体说是什么后遗症，但是目前症状比较轻的，可能没有后遗症，但是如果症状比较重，就可能出现呼吸困难，胸闷，发热，咳嗽等症状。  


<center><a href="https://clustrmaps.com/site/1bsr9"  title="Visit tracker"><img src="//www.clustrmaps.com/map_v2.png?d=j3DXKxFPrX7nY81AhJ7i6nPFDTA_q-gfSMhRR1rVS9c&cl=ffffff" /></a><center>