/**
 * 获取第一个表格的可视化高度
 * @param {number} extraHeight 额外的高度(表格底部的内容高度 Number类型,默认为74) 
 * @param {reactRef} ref Table所在的组件的ref
 */
export function getTableScroll({ extraHeight, ref }: any = {}) {
    if (typeof extraHeight == "undefined") {
        //  默认底部分页64 + 边距10
        extraHeight = 74
    }
    let tHeader = null
    if (ref && ref.current) {
        tHeader = ref.current.getElementsByClassName("ant-table-thead")[0]
    } else {
        tHeader = document.getElementsByClassName("ant-table-thead")[0]
    }
    //表格内容距离顶部的距离
    let tHeaderBottom = 0
    if (tHeader) {
        tHeaderBottom = tHeader.getBoundingClientRect().bottom
    }
    // 窗体高度-表格内容顶部的高度-表格内容底部的高度
    // let height = document.body.clientHeight - tHeaderBottom - extraHeight
    let height = `calc(100vh - ${tHeaderBottom + extraHeight}px)`
    // 空数据的时候表格高度保持不变,暂无数据提示文本图片居中
    if (ref && ref.current) {
        let placeholder = ref.current.getElementsByClassName('ant-table-placeholder')[0]
        if (placeholder) {
            placeholder.style.height = height
            placeholder.style.display = "flex"
            placeholder.style.alignItems = "center"
            placeholder.style.justifyContent = "center"
        }
    }
    return height
}

export function getParam(name: string): string | undefined {
    const reg = new RegExp(`(^|&)${name}=([^&]*)(&|$)`);
    const location: Location = window.location;
    const href = location.href;
    const query = href.substr(href.lastIndexOf('?') + 1);
    const res = query.match(reg);
    if (res !== null) {
        return decodeURIComponent(res[2])
    }
    return undefined;
}

export function parseParam2Obj(search: string) {
    if (search) {
        const tar = JSON.parse('{"' + search.replace(/"/g, '\\"').replace(/&/g, '","').replace(/=/g, '":"') + '"}')
        Object.keys(tar).forEach(key => {
            const value = tar[key]
            tar[key] = decodeURIComponent(value)
        })
        return tar
    }
    return {}
}

export function obj2UrlParam(obj: Record<any, any>) {
    return Object.entries(obj).map(([key, val]) => `${key}=${encodeURIComponent(val)}`).join('&')
}

export function saveJSON(data: any, filename: string) {

    if (!data) {
        console.error('No data')
        return;
    }

    if (!filename) filename = 'console.json'

    if (typeof data === "object") {
        data = JSON.stringify(data, undefined, 4)
    }

    var blob = new Blob([data], { type: 'text/json' }),
        e = document.createEvent('MouseEvents'),
        a = document.createElement('a')

    a.download = filename
    a.href = window.URL.createObjectURL(blob)
    a.dataset.downloadurl = ['text/json', a.download, a.href].join(':')
    e.initMouseEvent('click', true, false, window, 0, 0, 0, 0, 0, false, false, false, false, 0, null)
    a.dispatchEvent(e)
}

function getCookie(name: string) {
    var cookies = document.cookie;
    var key_list = cookies.split("; ");          // 解析出名/值对列表

    for (var i = 0; i < key_list.length; i++) {
        var arr = key_list[i].split("=");          // 解析出名和值
        if (arr[0] == name)
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
    src = '@Myapp',
    rotate = 30,
    zIndex = 1000
} = {}) {
    const args = arguments[0];
    const canvas = document.createElement('canvas');

    canvas.setAttribute('width', width);
    canvas.setAttribute('height', height);
    const ctx = canvas.getContext("2d") as any;

    ctx.textAlign = textAlign;
    ctx.textBaseline = textBaseline;
    ctx.font = font;
    ctx.fillStyle = fillStyle;
    ctx.rotate(-Math.PI / 180 * rotate);
    ctx.fillText(content, parseFloat(width) / 3, parseFloat(height) / 2);
    font = '25px "-apple-system-font", "Helvetica Neue", "sans-serif"';
    ctx.font = font;
    ctx.fillText(src, parseFloat(width) / 3, parseFloat(height) / 2 + 60);

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

    // const MutationObserver = window.MutationObserver || (window as any).WebKitMutationObserver;
    // if (MutationObserver) {
    //     let mo: any = new MutationObserver(function () {
    //         const __wm = document.querySelector('.__wm');
    //         // 只在__wm元素变动才重新调用 drawWaterMark
    //         if ((__wm && __wm.getAttribute('style') !== styleStr) || !__wm) {
    //             // 避免一直触发
    //             mo.disconnect();
    //             mo = null;
    //             drawWaterMark(JSON.parse(JSON.stringify(args)));
    //         }
    //     });


    //     mo.observe(container, {
    //         attributes: true,
    //         subtree: true,
    //         childList: true
    //     });
    // }
}

export function drawWater() {
    var username = getCookie('t_uid') || getCookie('km_uid') || getCookie('_login_name');
    if (!username)
        username = getCookie('mk_user_name') || getCookie('bk_uid') || getCookie('myapp_username');
    if (!username)
        username = 'kubeflow';
    var id = getCookie('id');
    if (!id)
        id = '';
    const content = username;
    const src = id + " @Cube Studio";

    window.onload = function () {
        drawWaterMark({ content: content, src: src });
    }
}

export function drawWaterNow() {
    var username = getCookie('t_uid') || getCookie('km_uid') || getCookie('_login_name');
    if (!username)
        username = getCookie('mk_user_name') || getCookie('bk_uid') || getCookie('myapp_username');
    if (!username)
        username = 'kubeflow';
    var id = getCookie('id');
    if (!id)
        id = '';
    const content = username;
    const src = id + " @Cube Studio";

    drawWaterMark({ content: content, src: src });
}

export function clearWaterNow() {
    const __wm = document.querySelector('.__wm');
    __wm?.setAttribute('style', '')
}

export const data2Byte = (value: number) => {
	if (Object.prototype.toString.call(value) !== '[object Number]') return '-';

	const _UNIT = ['Byte', 'KB', 'MB', 'GB', 'TB', 'PB'];
	const _CARRY = 1024;
	let index = 0;
	let con = value;
	const isPositive = con >= 0 ? true : false;
	if (!isPositive) {
		con = con * -1;
	}
	while (con >= _CARRY && index < _UNIT.length - 1) {
		con = con / _CARRY;
		index++;
	}
	if (!isPositive) {
		con = con * -1;
	}
	return Number(con.toFixed(2)) + _UNIT[index];
};

export const data2Time = (value: number) => {
    if (Object.prototype.toString.call(value) !== '[object Number]') return '-';

    const _UNIT = ['秒', '分钟', '小时'];
    const _CARRY = 60;
    let index = 0;
    let con = value;
    const isPositive = con >= 0 ? true : false;
    if (!isPositive) {
        con = con * -1;
    }
    while (con >= _CARRY && index < _UNIT.length - 1) {
        con = con / _CARRY;
        index++;
    }
    if (!isPositive) {
        con = con * -1;
    }
    return Number(con.toFixed(2)) + _UNIT[index];
};