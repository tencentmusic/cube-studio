import time, datetime, os

import pysnooper
# pip install pypandoc==1.12
# apt-get install -y texlive-full

class Markdown():

    def __init__(self):  # kubeconfig
        pass

    # 获取指定范围的pod
    # @pysnooper.snoop()
    def to_word(self,input_file, output_file):
        import pypandoc
        try:
            output = pypandoc.convert_file(input_file, 'docx', outputfile=output_file)
            assert output == ""
            print(f"成功将 {input_file} 转换为 {output_file}")
        except Exception as e:
            print(f"转换过程中出现错误: {e}")

    def to_html(self,input_file, output_file):
        import pypandoc
        try:
            output = pypandoc.convert_file(input_file, 'html', outputfile=output_file)
            assert output == ""
            print(f"成功将 {input_file} 转换为 {output_file}")
        except Exception as e:
            print(f"转换过程中出现错误: {e}")

    # @pysnooper.snoop()
    def to_pdf(self,input_file, output_file):
        import markdown
        import pdfkit

        with open(input_file, 'r', encoding='utf-8') as f:
            md_content = f.read()

        html_content = markdown.markdown(md_content,output_format='html')

        # 添加 CSS 样式以支持中文
        html_content = f'''
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: "Microsoft YaHei", "微软雅黑", "宋体", sans-serif;
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
        '''

        pdfkit.from_string(html_content, output_file)

if __name__=='__main__':
    mark = Markdown()
    mark.to_pdf('markdown/test.md','markdown/test.pdf')


