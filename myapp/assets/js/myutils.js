
// 转到携带参数的新网址
function to_new_url(attr,value) {
    var url = document.location.toString();
    var arrUrl = url.split("?");
    var path =  arrUrl[0];
    var new_url='';
    if(arrUrl.length>1){
        var para = arrUrl[1];
        var vars = para.split("&");

        var new_para='';
        exist=false;
        for (var i=0;i<vars.length;i++) {
            if(i>0)
                new_para+="&";
            var pair = vars[i].split("=");
            new_para+=pair[0];
            if(pair[0] == attr){
                new_para+='='+value;
                exist=true;
            }
            else{
                new_para+='='+pair[1];
            }
        }

        if(!exist){
            new_para+='&'+attr+'='+value
        }
        new_url=path+"?"+new_para
    }else{
        new_url=path+"?"+attr+'='+value
    }
    console.log(new_url);
    window.open(new_url,'_self');
}


function set_change(attr) {
    var myselect=document.getElementById(attr);
    var new_text=myselect.options[myselect.selectedIndex].text;
    to_new_url(attr,new_text)
}

