/**
 * @description 函数防抖
 * @param fn 函数
 * @param delay 间隔时间
 * @returns {Function}
 */
export function debounce<F extends (...params: any[]) => void>(fn: F, delay: number): F {
  let timer: any = null;
  return function (this: any, ...args: any[]) {
    clearTimeout(timer);
    timer = window.setTimeout(() => fn.apply(this, args), delay);
  } as F;
}

/**
 * @description 校验是否为 JSON 字符串
 * @param str 字符串
 * @returns {Boolean}
 */
export function isJsonString(str: string): boolean {
  try {
    JSON.parse(str);
    return true;
  } catch (error) {
    return false;
  }
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
