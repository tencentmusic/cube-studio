/**
 * @func
 * @desc 一个带参数的函数
 * @param {string} name -  要查询的参数
 * @param {string} url -  被查询的url
 * @param {booleam} bool -  默认为true
 * @return 返回值
 */
function getDetailed(name: string, str: string) {
  const obj: any = {};

  if (name !== '') {
    const reg = new RegExp('(^|&)' + name + '=([^&]*)(&|$)', 'i');
    const r = str.match(reg);

    if (r !== null) {
      obj[name] = r[2];
    } else {
      obj[name] = '';
    }
  } else {
    // 查询所有的参数
    const arr = str.split('&');

    arr.forEach(element => {
      const temp = element.split('=');

      obj[temp[0]] = temp[1];
    });
  }
  return obj;
}
const getParamter = function (name: string, url: string, bool = false): any {
  url = decodeURIComponent(url || window.location.href);
  const answer: any = {};
  const indexBefore = url.indexOf('?'); // search字段的'?'位置
  let end = url.indexOf('#');

  if (end === -1) {
    // 判断url中是否有hash
    end = url.length;
  }

  // 可以取hash字段
  if (end !== url.length && bool) {
    let strAfter = url.substring(end + 1); // hash字段
    const indexAfter = strAfter.indexOf('?'); // hash字段的'?'位置

    if (indexAfter !== -1) {
      strAfter = strAfter.substring(indexAfter + 1);
      Object.assign(answer, getDetailed(name, strAfter));
    }
  }
  // 可以取search字段
  if (indexBefore !== -1) {
    const strBefore = url.substring(indexBefore + 1, end); // search字段

    Object.assign(answer, getDetailed(name, strBefore));
  }
  return answer;
};

export default getParamter;
