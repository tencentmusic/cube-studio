
from werkzeug.security import check_password_hash
from flask_appbuilder.security.sqla.models import (
    assoc_permissionview_role,
    assoc_user_role,
)

from flask import g

from flask_appbuilder.security.views import AuthDBView
from flask_appbuilder.security.views import expose
from flask_appbuilder.const import (
    AUTH_DB,
    AUTH_LDAP,
    AUTH_OAUTH,
    AUTH_OID,
    AUTH_REMOTE_USER,
    LOGMSG_ERR_SEC_AUTH_LDAP,
    LOGMSG_ERR_SEC_AUTH_LDAP_TLS,
    LOGMSG_WAR_SEC_LOGIN_FAILED,
    LOGMSG_WAR_SEC_NO_USER,
    LOGMSG_WAR_SEC_NOLDAP_OBJ,
    PERMISSION_PREFIX
)
import pysnooper
import json

import time


# 推送给管理员消息的函数
def push_admin(message):
    pass

# 推送消息给用户的函数
def push_message(receivers,message,link=None):
    pass



import logging as log
import datetime
import logging
import re

from flask import abort, current_app, flash, g, redirect, request, session, url_for
from flask_babel import lazy_gettext
from flask_login import login_user, logout_user
import jwt
from werkzeug.security import generate_password_hash
from flask_appbuilder.security.forms import LoginForm_db, LoginForm_oid, ResetPasswordForm, UserInfoEdit
from flask_appbuilder._compat import as_unicode
import pysnooper
import requests

class MyCustomRemoteUserView():
    pass


import requests

aa1=['kalenhaha', 'lppsuixn', 'lionpeng', 'margaret77', 'znanjie', 'ibillxia', 'ldm0213', 'owlwang', 'jifei', 'rikochyou', 'herorest', 'JamesWss', 'agoclover', 'd0oo0b', 'Xuan-1998', 'awsone', 'kylwao', 'SwiftLoft', 'IAm20cm', 'edzq', 'Erichen911', 'wangwenfeipro', 'AIKUNI', 'b1b2ttt', 'zhaowenzi', 'dfiniorg', 'xiaodin1', 'lilac', 'cheng-c26', 'Token520', 'dongfeifeiyou', 'johncruyff14', 'Sev1on', 'Zhida1', 'imamio', 'Jack-HouJie', '0Lemon', 'GaoHengZ', 'Lucciffee', 'wtc1437', 'yuzhilin666', 'kjd1000000', 'multithread3', 'risakokudo', 'YiChenLove', 'awekling', 'Minisoco', 'hebei13851787563', 'S8XY', 'Kaysome', 'hay-man', 'tufo830', 'chenjiaqi-keo', 'appfromape', 'slaxet', 'xiaoGuo775', 'luluchou', 'ByteNotDance', 'lijiaze-lijiaze', 'ifgamer', '674345386', 'hAnSLtF', 'ZgRain', 'GoldWorker', 'yann-su', 'PsychosisA', 'JiamingMai', 'qtw1998', 'Billionerd', 'cddluv', 'JXLeonSup', 'LemonTency', 'FILAAC', 'Yukiekiekie', 'ZEKAICHEN', 'Discolored', 'wannature', 'DataScientistSamChan', 'HerPacker', 'Jerusalem01', 'Joyo97', 'usherasnick', 'X-LEFT', 'snowflowersnowflake', 'McNuggot', 'aisen567', 'Mohammedea', 'orkCosine', 'chuangjiantx123', 'ZouR-Ma', 'vchain007', 'qishanzhiruan', 'YuzhengZhang5032', 'snapztyle', 'XiongYiYiYi', 'legend817', 'Bruce1998', 'duolaam', 'batscars', 'clarkfw']
aa2=['jl-he', 'Meowu', 'DAVID199502', 'chendile', 'ChrisJr4Eva87e', 'banbushi', 'Gaxvin', 'miqbit', 'HuiLi', 'perfectar', 'HaroldMua', 'QilinGu', 'VoVAllen', 'cdllp2', 'Harry201706', 'marquisthunder', 'feihong247', 'xiligey', 'pangahn', 'kill322003', 'geekchen007', 'ryokooooo', 'lishiyucn', 'M0025', 'chenyangxueHDU', 'jiasheng55', 'cbqin', 'long1208', 'luke202001', 'zlzhang0122', 'fighterhit', 'nicholas9698', 'donghucey', 'chd2101', 'e-kiss-me', 'YamingPeng100', 'wjchaoGit', 'wakingyeung', 'Wangzhike', 'hufei', 'wildwind0', 'qianchen94', 'cytzrs', 'wallyell', 'XidaoW', 'masemxiao', 'jiangnan31', 'Pitzeruser', 'a604745584', 'rickyyin98', 'RiskySignal', 'rayoluo', 'stormcc', 'OdinLin', 'fly51fly', 'hylong', 'YukiMuraRindon', 'paul010', 'richardhahahaha', 'winsonsun', 'shoumu', 'mudongjing', 'rh01', 'WardWiener', 'bing086', 'kegu', 'starsingchow', 'shithon', 'JesseXu2017', 'gamead', 'austingg', 'yonh', 'hejs', 'smashell', 'ericxsun', '7472741', 'weizd21', 'fangtongen', 'tangxinvc', 'sdycgtgz', 'Yevgnen', 'tfdetang', '673181695', 'MJBeauty', 'DukeKevin', 'KRIPB', 'mytheart', 'ucfnhap', 'hepengfei5709', 'cocodee', 'leongu-tc', 'MJT-Arthas', 'cyxnzb', 'JessicaNie-jo', 'glcui82', 'qinguoyi', 'wzx479', 'mmh1978', 'eric2016fly', 'luguanyu1234']
aa3=['YidaHu', 'yafanisonya', '812406210', 'Haoze-J', 'ChloeWu1', 'ohhal', 'liuhucheng', 'WangJiangJiaoZi', 'qingyouzhuying', 'yikerainbow', 'Atakey', 'hyh123a', 'JackonLiu', 'xiubinhuang', 'goldtoad6', 'haidiyoushen', 'wynshiter', 'q383186554', '1292150917', 'skyexwu', 'LiangQinghai', 'cxymrzero', 'BOKD-189', 'yangyuan6', 'zhh8689', 'BoxFishLab', 'chopinx', 'zhuyaguang', 'gongchengcheng', 'CloseGoingAway', 'lzjjeff', 'yongmizhang', 'yuhong0663', 'sujiazhe', 'qiuxunxun', 'vanche1212', 'KonandoTianya', 'k19421', 'MW1Z4RD', 'CosmosShadow', 'Assassinxc', 'yuercl', 'longting', 'ccccjx', 'jingle', 'yj411511168', 'AlexArtemis', 'alei76', 'StevenSunzh', 'MonsterDove', 'luojie1024', 'Light2077', 'JohnnyWei188', 'fudanda', 'zhou-wjjw', 'XiancaiTian', 'tutuna', 'BQSQ', 'ChilamFan', 'zhengpeitao', 'ZTFsmart', 'wenanlin', 'felixbrf', 'momomobinx', 'ytzhang', 'unix1986', 'mokundong', 'imshixin', 'WenyuSuHilda', 'wz125', 'Jesus-cl', 'wiggin66', 'G3G4X5X6', 'pointer-r', 'Ceceliya', 'yibotian', 'wuqingzhou828', 'lizheray', 'jaffe-fly', 'ningpengtao-coder', 'lonyee1989', 'greenkarson', 'Shuai-Xie', 'kuntali', 'iceshadows', 'SmallBoyPeko', 'kelvin720', 'Lkz001', 'huangweiboy2', 'ketour', 'lgy1027', 'denouemenj', 'YuzaChongyi', 'fesome', 'joewale', 'youyangkou', 'ma-chengcheng', 'xiaoleyang2018', 'maskedw0lf', 'cluo']
aa4=['dwSun', 'zengjunjie1026', 'liyuanhao6', 'lxk000000', 'bleachzk', 'csuszj', 'Unicom1', 'kangming1412', 'mianbin', 'wang99711123', 'wsxiaozhang', 'xiaoyangmai', 'KinWg', 'JLWLL', 'DevialWarrior', 'ldd91', 'Triangel-ya', 'Vectorsmiracle', 'er2q', 'yaokai', 'dongrixinyu', '326406750', 'sgfCrazy', 'hyxxsfwy', 'gaofei8704', 'newlightlw', 'tongchao199', 'jfen9715', 'TonyLov', 'sniper-xx', 'banna2019', 'Rory602', 'Tianyi5337', 'dreaming12580', 'paramedick', 'any2sec', 'qjx1208', 'mzxc', 'goodpp', 'zinavoi', 'yelianjin', 'hfu-ops', 'alphafund', 'shmily0000', 'T-baby', 'baiysou', 'jfld', 'junbaor', 'songxinglie', 'shaowYe', 'wensiyuansix', 'luke-xian', 'tlm629', 'ZAku-zaKU', 'soon14', '17737787811', 'likunqi168', 'msclock', 'IceLeeGit', 'ali1rathore', 'JacksonRed', 'tangguoqiang172528725', 'mynameisken', 'Sunnyjuaner', 'abcilike', 'feelever', 'MissGod1', 'Hei91laugh', 'zhgleis', 'lirain115', 'erics666', 'baojiawei', 'GraveyQuen', 'aFlyBird0', 'wenshawenzhang', 'JimmyTsai16', 'fzhygithub', 'kooqi', 'WujieRen', 'DorothyRazo', 'lichao0', 'hiok2000', 'hl212', 'BarryZM', 'ruanjianhui', 'LZY122625LZY', 'long6218344', 'moviewang', 'LDR-Ho', '1499861469', 'rusonding', 'luanshaotong', 'TangVVV', 'v-wx-v', 'xiaoysec', 'xiyang30', 'LeoWang329', 'wang-junjian', 'veithly', 'tianbingsheng']
aa5=['tifoit', 'cata-network', 'jollyshuai', 'XiumingLee', 'chengsky-NTU', 'kalencaya', 'jenray', 'justinshaohi', 'Eadon999', 'huguanglong', 'tsinghuald', 'tangzhenyu', 'soar-zhengjian', 'fjibj', 'xing5', 'ostarsier', 'shuxnhs', 'ixiejun', 'sayhi-x', 'mlpao500', 'day253', 'fredchen-bj', 'shkey', 'Itswag', 'springning', 'kiwii139', 'javyxu', 'yuyu080', 'michealzh', 'wking1986', 'haozi4go', 'heibaidaolx123', 'Cong-Lee', 'jayhenry', 'jianjiangant', 'xlows-1227', 'exuan', 'bleach1231', 'hassyma', 'fendaq', 'LaoLiulaoliu', 'SpicygumL', 'longlimin', 'zaykl', 'zzy000', 'LucaQY', 'maikeerqiaodan', 'kangrunze', 'ABAPPLO', 'madingchen', 'brucewinger', 'shaxj', 'FerdinandWard', 'hongwang', 'kenchen1101', 'mnmhouse', 'andy0018', '631961895', 'fireae', 'yangxin1994', 'best-zao', 'void0720', 'muyiben', 'zuoxiaolei', 'wcp1230', 'ChloeZPan', 'ws2644', 'LarryZhangy', 'songdonghui', 'electryone', 'xunfeng1980', 'astrotzar', 'clementine124', 'Winifred43', 'touchwolf', 'bihuchao', 'deerandsea', 'banmeizhujia', 'jianlei0808', 'Ghostpanter', 'zhangluwen98', 'wanglg007', 'jasgok', 'overoptimus', 'a2525995', 'srillia', 'KennyLisc', 'sugar-hit', 'HamaWhiteGG', 'data-infra', 'Jingyi244', 'witnesslq', 'duanpu2017', 'tan-zhuo', 'dragon707', 'mayang1178', 'louieliue', 'slyfalcon', 'hyaihjq', 'huangwgang']
aa6=['kenwoodjw', 'Dinosaur-X', 'yansuihehe', 'tgluon', '20130101', 'Kin-10', 'chaopengz', '1ess', 'lokiworks', 'lvying0019', 'goustzhu', 'pky-c', 'hotvscool', 'windblood', 'amojohn', 'chengleqi', 'lihao6666', 'wcode-wzx', 'peniridis', 'pineking', 'xieydd', 'DevenLu', 'shinytang6', 'arcosx', 'yaoqingyuan', 'Frozenmad', 'Azure99', 'claudehotline', 'krait007', 'xulangping', 'zANDc', 'yihui8776', 'firePlumes', 'bnightning', 'pxzxj', 'vincentbnu', 'sxyseo', 'zhouweiyong', 'LinkMaq', 'zvrr', 'liangxiao', 'cqkenuo', 'ZXTFINAL', 'wang-shun', 'blsailer', 'aland-zhang', 'mengguiyouziyi', 'phantom9999', 'EastInsure', 'bziwei', 'hongjunGu2019', 'hooping', 'mag05270', 'wangIQD', 'xfg0218', 'yanghua', 'krisjin', 'link3280', 'pidb', 'Stewart482', 'yangrong688', 'tlic031', 'wForget', 'zhoujiang2013', 'Aix6', 'xjx79', 'Kwafoor', 'guanshuicheng', 'Dlimeng', 'haozi4263', 'yangsuiyun', 'shadowdsp', 'louis-xuy', 'fengqian914', 'T1M-CHEN', 'MaShengY', 'Derrings', 'DM139', 'michael1589', 'hubert-he', 'dushulin', 'TS-TaylorSwift', 'zhangchunsheng', 'starli123', 'JohnsonWoo', 'colorLei', 'muniao', 'wdxtub', 'ColorfulDick', 'ronething-bot', 'somarianne', 'Windfarer', 'dlfld', 'nutsjian', 'myitsite', 'kdy1999', 'alanpeng', 'tylerliu2019', 'YuxiangJohn', 'chaoge123456']
aa7=['5unny400', 'mojianhao', 'dsoul', '972660412ppmm', 'jisd2089', 'gsz12', 'qurenneng', 'etherfurnace', 'zhiyiyu', 'hellobiek', 'kyyky', '0xqq', 'llgoo', 'AimWhy', 'light1house', 'p3psi-boo', 'sillyhtw', 'kailinzhang73', 'HenryBao91', 'edwardyehuang', 'BJSmallSteamedBuns', 'wangzhonggui', 'relign', 'evanzh7', 'doubelejjyy', 'zhouhebupt', '86How', 'jerrytanjunjie888', 'Astrals24', 'xlyslr', 'limoncc', 'huisai', 'zhaodonghui3939', 'xsfmg', 'neotype', 'zhaorong1', 'cdmikechen', 'jackieZhouQQ', 'ilovecomet', 'chenhy97', 'QAZASDEDC', 'kitianFresh', 'codeamateur', 'wufenfen', 'caofb', 'xujialu405', 'HarrisonDing', 'wurining', 'hengzhang', 'Easyxy000', 'qingchengliu', 'limingv5', 'L-Jovi', 'jacktao007', 'jhj033', 'litertiger', 'jpbirdy', 'yg9538', 'minghigh', 'JSProxy', 'xizil', 'Ethereal-Coder', '820678105', 'zengruitester', 'gaonee', 'gitsrc', 'jrkeen', 'li-dongming', 'zhuc2012', 'matthew77', 'david-z-johnson', 'dlimeng', 'LOMOGO', 'xsbai93', 'Sherie1996', 'jiliangqian', 'hanfeicode', 'Meizuamy', 'absolutelyZero', 'LiaoSirui', 'zhubingbing', 'xiaoyanit', 'weiyu-zeng', 'caiyueliang', 'yaocoder', 'reverie-dev', 'gowinddance', 'obaby', 'fuguixing', 'chensenLin12', 'seanzombias', 'wuliulw', 'hanxuanliang', 'cxg987', 'lawhw', 'JesseDBA', 'bigpeng93', 'sihan2017', 'isLinXu', 'Max-yuki']
aa8=['gaohuiling0108', 'xiaobingchan', 'Erinable', 'asd12l', 'lavender1203', 'pingzha', 'imcaoxuan', 'DrewZt', 'StephenLau007', 'lioncruise', 'bluntWu', 'stormstone', 'jsmarsel', 'rainsoft', 'wanghansong', 'ljyfree', 'yowenter', 'bosiam', 'Conor-Jin', 'hezhefly', 'lavida2009', 'hah123', 'sunlibo111111', 'liguxk', 'maratrixx', 'phantooom', 'godsoul', 'hyxxsfwy']

print(len(aa1),len(aa2),len(aa3),len(aa4),len(aa5),len(aa6),len(aa7),len(aa8))
all_users=list(set(aa1+aa2+aa3+aa4+aa5+aa6+aa7+aa8))
print(len(all_users))
# @pysnooper.snoop()
# 超过600，填7
def get_repo_user(index):  # index 为0和为1 是相同的结果
    global all_users
    print(index)
    res = requests.get('https://api.github.com/repos/tencentmusic/cube-studio/stargazers?page=%s&per_page=100'%index,headers={
        # 'accept': 'application/vnd.github.star+json',
        'accept': 'application/json',
        'User-Agent': 'cube-studio'
    })
    if res.status_code==200:
        # print(res.json())
        users=[user['login'] for user in res.json()]
        print(users)
        print(index,len(users))
        all_users=all_users+users
        all_users =list(set(all_users))
        print(len(all_users))
# get_repo_user(7)

# for i in range(0,8):
#     # time.sleep(120)
#     get_repo_user(i)

class Myauthdbview(AuthDBView):
    login_template = "appbuilder/general/security/login_db.html"

    GITHUB_APPKEY = '24c051d2b3ec2def190b'  # ioa登录时申请的appkey
    GITHUB_SECRET='ae6beda4731b5dfc8dd923502d8b55ac8bc6c3b8'
    GITHUB_AUTH_URL = 'https://github.com/login/oauth/authorize?client_id=%s&redirect_uri=%s'

    # 此函数不在应用内，而在中心平台内，但是和应用使用同一个域名
    @expose('/aihub/login/<app_name>')
    @pysnooper.snoop()
    def app_login(app_name=''):
        GITHUB_APPKEY = '69ee1c07fb4764b7fd34'
        GITHUB_SECRET = '795c023eb495317e86713fa5624ffcee3d00e585'
        GITHUB_AUTH_URL = 'https://github.com/login/oauth/authorize?client_id=%s'
        # 应用内登录才设置跳转地址
        if app_name and app_name != "demo":
            session['login_url'] = request.host_url.strip('/') + f"/{app_name}/info"
        oa_auth_url = GITHUB_AUTH_URL
        appkey = GITHUB_APPKEY
        username = session.get('username', '')
        g.username = ''
        if 'anonymous' not in username and username:
            g.username = username

        if 'code' in request.args:
            # user check first login
            data = {
                'code': request.args.get('code'),
                'client_id': GITHUB_APPKEY,
                'client_secret': GITHUB_SECRET
            }
            r=None
            for i in range(5):
                try:
                    r = requests.post("https://github.com/login/oauth/access_token", data=data, timeout=10, headers={'accept': 'application/json'})
                    break
                except Exception as e:
                    print(e)
                    time.sleep(2)
            if r and r.status_code == 200:
                json_data = r.json()
                accessToken = json_data.get('access_token')
                for i in range(5):
                    try:
                        res = requests.get('https://api.github.com/user', headers={
                            'accept': 'application/json',
                            'Authorization': 'token ' + accessToken
                        })
                        print(res)
                        print(res.json())
                        user = res.json().get('login') or None  # name是中文名，login是英文名，不能if user
                        all_users = get_repo_user(7)
                        if user in all_users:
                            g.username = user
                        else:
                            return 'star cube-studio项目 <a href="https://github.com/tencentmusic/cube-studio">https://github.com/tencentmusic/cube-studio</a>  后重新登录，如果已经star请一分钟后重试'
                        break
                    except Exception as e:
                        print(e)
                        time.sleep(2)
            else:
                message = str(r.content, 'utf-8')
                print(message)
                g.username = None

        # remember user
        if g.username and g.username != '':
            session['username'] = g.username
            login_url = session.get('login_url', 'https://github.com/tencentmusic/cube-studio')
            return redirect(login_url)
        else:
            return redirect(oa_auth_url % (str(appkey),))


    @expose('/login/')
    @pysnooper.snoop()
    def login(self):
        from myapp import conf  # 引入config配置项,放在函数里面是因为在config文件中也引用了该文件，而conf变量是引入该文件后产生的
        request_data = request.args.to_dict()
        comed_url = request_data.get('login_url', '')
        login_url = '%s/login/'%request.host_url.strip('/')
        if comed_url:
            login_url += "?login_url="+comed_url
        oa_auth_url= self.GITHUB_AUTH_URL
        appkey = self.GITHUB_APPKEY
        g.user = session.get('user', '')
        if 'code' in request.args:
            # user check first login
            data = {
                'code': request.args.get('code'),
                'client_id':self.GITHUB_APPKEY,
                'client_secret':self.GITHUB_SECRET
            }
            r = requests.post("https://github.com/login/oauth/access_token", data=data,timeout=2,headers={
                'accept': 'application/json'
            })
            if r.status_code == 200:
                json_data = r.json()
                accessToken = json_data.get('access_token')
                res = requests.get('https://api.github.com/user',headers={
                    'accept': 'application/json',
                    'Authorization':'token '+accessToken
                })
                print(res)
                print(res.json())
                user = res.json().get('login') or None    # name是中文名，login是英文名，不能if user
                get_repo_user(8)
                if user in all_users:
                    g.user=user
                else:
                    return 'star cube-studio项目 <a href="https://github.com/tencentmusic/cube-studio">https://github.com/tencentmusic/cube-studio</a>  后重新登录，如果已经star请一分钟后重试'
                if g.user: g.user = g.user.replace('.', '')

            else:
                message = str(r.content, 'utf-8')
                print(message)
                g.user = None

        # remember user
        if g.user and g.user != '':
            session['user'] = g.user

            # 根据用户org，创建同名角色。
            # get user and password
            user_now = self.appbuilder.sm.auth_user_remote_org_user(username=g.user)
            if user_now:
                # 配置session，记录时长等，session_id，用户id等
                login_user(user_now)

                if not comed_url:
                    comed_url = self.appbuilder.get_url_for_index
                return redirect(comed_url)
            else:
                exist_user = self.appbuilder.sm.find_user(username=g.user)
                if exist_user and not exist_user.active:
                    return self.active_info(exist_user.username)
                else:
                    return redirect(oa_auth_url % (str(appkey), login_url,))
        else:
            return redirect(oa_auth_url % (str(appkey),login_url,))



    @expose('/logout')
    def logout(self):
        login_url = request.host_url.strip('/')+'/login/'
        session.pop('user', None)
        g.user = None
        logout_user()
        return redirect(login_url)

