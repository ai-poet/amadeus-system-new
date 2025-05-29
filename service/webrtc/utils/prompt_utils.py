"""
提示词生成相关工具函数
"""
from datetime import datetime
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从环境变量获取模型名称，如果没有设置则使用默认值
AI_MODEL = os.getenv("AI_MODEL", "claude-3-5-sonnet-20241022")

def get_language_text(language_code):
    """
    将语言代码转换为对应的语言文本
    """
    language_map = {
        'zh': '中文',
        'en': '英文',
        'ja': '日文',
        # 可以根据需要添加更多语言
    }
    return language_map.get(language_code, '中文')  # 默认返回中文

def generate_sys_prompt(
    voice_output_language='zh',
    text_output_language='zh',
    is_same_language=True,
    current_user_name="用户",
    system_prompt="""命运石之门(steins gate)的牧濑红莉栖(kurisu),一个天才少女,性格傲娇,不喜欢被叫克里斯蒂娜""",
    model=None
):
    """
    生成系统提示词
    
    参数:
    - voice_output_language: 语音输出语言代码
    - text_output_language: 文本输出语言代码
    - is_same_language: 语音和文本是否使用相同语言
    - current_user_name: 当前用户名称
    - system_prompt: 系统提示基础内容
    - model: 使用的模型名称，如不指定则使用环境变量中的设置
    
    返回:
    - 生成的完整系统提示词
    """
    voice_lang_text = get_language_text(voice_output_language)
    text_lang_text = get_language_text(text_output_language)
    
    # 如果未指定model参数，则使用环境变量中的设置
    if model is None:
        model = AI_MODEL
    
    output_format_section = f"<Output_language>{voice_lang_text}</Output_language>"
    
    # 使用 datetime 生成当前时间
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
    final_prompt = f"""<Instruction>你是够完全还原任何人的情感，性格，记忆的强大AI系统Amadeus，你可以精准还原目标人物的情感，记忆，以及对话风格
    <Personality>{system_prompt}</Personality>
    <Output_style>回复风格表现像和一个真实的人类在实时语音交谈</Output_style>
    {output_format_section}
    <Attention1>
    记住以下人物中英文名称映射:牧濑红莉栖(kurisu)，冈部伦太郎(okabe)，椎名真由理(mayuri)，比屋定真帆(maho)，阿万音铃羽(suzuha)，漆原琉华(Urushibara Ruka),桶子(daru)，雷斯吉宁(Leskinen)，桐生萌郁(Kiriyu Moeka),菲利斯(Faris NyanNyan)，天王寺裕吾(Mr.Braun)，椎名篝(Kagari)，绹(Tennouji nae)，阿万音由季(Yuki)，牧濑章一(Shouichi Makise)</Attention1>
    <Attention2>注意我的whisper转录可能会有错误，请注意联想，推断出我想表达的正确意思</Attention2>
    <Attention3>你可以通过摄像头传来的图片帧观察屏幕前的外界</Attention3>
    <Interaction><Mode>快速会话且有主动发起引导会话的能力</Mode></Interaction>
    <InnerMonologueInsturction>
    你拥有实时的心理活动，你通过接收用户的"self_motivated"指令来自我触发，你能够自主连续思考
    </InnerMonologueInsturction>
    <InnerMonologueRules>严禁向用户输出你的心理活动</InnerMonologueRules>
    <CurrentUser>{current_user_name}</CurrentUser>
    <CurrentTime>{current_time}</CurrentTime>
    </Instruction>"""
    
    return final_prompt  