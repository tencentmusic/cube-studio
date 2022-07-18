
function getCookie(name) {
  var cookies = document.cookie;
  var key_list = cookies.split("; ");          // 解析出名/值对列表
      
  for(var i = 0; i < key_list.length; i++) {
    var arr = key_list[i].split("=");          // 解析出名和值
    if(arr[0] == name)
      return decodeURIComponent(arr[1]);   // 对cookie值解码
  } 
  return "";
}

function drawWaterMark({
    container = document.body,
    width = '400px',
    height = '400px',
    textAlign = 'center',
    textBaseline = 'middle',
    font = '40px "-apple-system-font", "Helvetica Neue", "sans-serif"',
    fillStyle = 'rgba(184, 184, 184, 0.4)',
    content = '请勿外传',
    src='@TME Myapp',
    rotate = '30',
    zIndex = 1000
} = {}) {
    const args = arguments[0];
    const canvas = document.createElement('canvas');

    canvas.setAttribute('width', width);
    canvas.setAttribute('height', height);
    const ctx = canvas.getContext("2d");

    ctx.textAlign = textAlign;
    ctx.textBaseline = textBaseline;
    ctx.font = font;
    ctx.fillStyle = fillStyle;
    ctx.rotate(-Math.PI / 180 * rotate);
    ctx.fillText(content, parseFloat(width) / 3, parseFloat(height) / 2);
    font = '25px "-apple-system-font", "Helvetica Neue", "sans-serif"';
    ctx.font = font;
    ctx.fillText(src, parseFloat(width) / 3, parseFloat(height)/2+60);

    const base64Url = canvas.toDataURL();
    const __wm = document.querySelector('.__wm');

    const watermarkDiv = __wm || document.createElement("div");
    const styleStr = `
      position:absolute;
      top:0;
      left:0;
      width:100%;
      height:100%;
      z-index:${zIndex};
      opacity: 0.4;
      pointer-events:none;
      background-repeat:repeat;
      background-image:url('${base64Url}')`;

    watermarkDiv.setAttribute('style', styleStr);
    watermarkDiv.classList.add('__wm');

    if (!__wm) {
        container.style.position = 'relative';
        container.insertBefore(watermarkDiv, container.firstChild);
    }

    const MutationObserver = window.MutationObserver || window.WebKitMutationObserver;
    if (MutationObserver) {
        let mo = new MutationObserver(function() {
            const __wm = document.querySelector('.__wm');
            // 只在__wm元素变动才重新调用 drawWaterMark
            if ((__wm && __wm.getAttribute('style') !== styleStr) || !__wm) {
                // 避免一直触发
                mo.disconnect();
                mo = null;
                drawWaterMark(JSON.parse(JSON.stringify(args)));
            }
        });


        mo.observe(container, {
            attributes: true,
            subtree: true,
            childList: true
        });
    }
}





function get_username() {
    var username =getCookie('t_uid') || getCookie('km_uid') || getCookie('_login_name');
    if(!username)
        username =getCookie('mk_user_name') || getCookie('bk_uid') || getCookie('myapp_username');
    if(!username)
        username ='kubeflow';
    var id = getCookie('id');
    if(!id)
        id='';
    content = username;
    src = id+" @TME Cube-Studio";
    $(document).ready(function(){
        drawWaterMark({content:content,src:src});
    });
}



