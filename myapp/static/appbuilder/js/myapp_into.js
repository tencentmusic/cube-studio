// 注册message消息监听
window.addEventListener('message', function (event) {
  const data = event.data || {};
  switch (data.type) {
    case 'fullview':
      const target = data.target;
      // console.log(document.querySelector(target).style.cssText);
      document.querySelector(target).style.cssText += `position: fixed;
        top: 0px;
        left: 0px;
        width: 100%;
        height: 100%;
        z-index: 999;
        background: white;`;
      break;
  }
});