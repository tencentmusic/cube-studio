import { message } from 'antd';

export function copyStr(txt) {
    console.log(txt)
    if (!txt) return false
    const input = window.document.createElement("input")
    window.document.body.appendChild(input);
    input.value = txt
    input.select()
    window.document.execCommand("Copy")
    input.remove()
    message.success("复制成功")
}