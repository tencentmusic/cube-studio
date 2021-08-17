/**
 * 函数防抖
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
