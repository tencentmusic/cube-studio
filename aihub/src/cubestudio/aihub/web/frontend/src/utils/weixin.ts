import axios from '../api/index'
const wx = require('weixin-js-sdk')

// TODO 
const wxSignUrl = 'http://www.data-master.net/wechat/jsapi';

export interface ShareContext {
  title?: string
  link?: string
  desc?: string
  imgUrl?: string
}

const isInWeixin = function (): boolean {
  var ua = navigator.userAgent.toLowerCase()
  var isWeixin = ua.indexOf('micromessenger') != -1
  if (!isWeixin) {
    // document.head.innerHTML = '<title>抱歉，出错了</title><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=0"><link rel="stylesheet" type="text/css" href="https://res.wx.qq.com/open/libs/weui/0.4.1/weui.css">';
    // document.body.innerHTML = '<div class="weui_msg"><div class="weui_icon_area"><i class="weui_icon_info weui_icon_msg"></i></div><div class="weui_text_area"><h4 class="weui_msg_title">请在微信客户端打开链接</h4></div></div>';
    return false
  }
  return true
}

const weixin = function (): Promise<any> {
  return new Promise((resolve, reject) => {
    let url = window.location.href

    axios.get(wxSignUrl, {
      params: {
        url: url
      }
    }).then((ress: any) => {
      let res = ress.data
      if (res.code == 0) {
        wx.config({
          debug: true,
          appId: res.data.appId,
          timestamp: res.data.timestamp,
          nonceStr: res.data.nonceStr,
          signature: res.data.signature,
          jsApiList: [
            'showOptionMenu',
            "hideMenuItems",
            "showMenuItems",
            'onMenuShareTimeline',
            'onMenuShareAppMessage',
            'updateAppMessageShareData', // 自定义“分享给朋友”及“分享到QQ”按钮的分享内容
            'updateTimelineShareData', // 自定义“分享到朋友圈”及“分享到QQ空间”按钮的分享内容（1.4.0）
          ]
        })
        wx.ready((res: any) => {
          // @ts-ignore
          resolve(wx, res)
        })
        wx.error((err: any) => {
          // @ts-ignore
          reject(wx, err)
        })
      }
    })
  })
}

// 微信分享
const share = function (share: ShareContext): void {
  weixin().then((wx) => {
    wx.ready(() => {
      setTimeout(() => {
        wx.updateAppMessageShareData({
          title: share.title,
          link: share.link,
          desc: share.desc,
          imgUrl: share.imgUrl,
          type: 'link',
          success: function () {
            alert('分享成功')
          },
          cancel: function () {
            alert('分享失败')
          }
        })
      }, 500)
    })
  })
}

export {
  isInWeixin,
  weixin,
  share
}
