(this.webpackJsonpweb=this.webpackJsonpweb||[]).push([[0],{193:function(t,e,n){},285:function(t,e,n){},290:function(t,e){},291:function(t,e){},299:function(t,e){},302:function(t,e){},303:function(t,e){},306:function(t,e,n){"use strict";n.r(e);var a=n(54),c=n(17),r=n.n(c),i=n(65),s=n.n(i),u=(n(285),n(193),n(4)),o=n.n(u),d=n(19),l=n(11),f=n(6),j=n(191),h=n(257),b=(n(305),function(){var t,e,n=Object(c.useState)(null),r=Object(f.a)(n,2),i=r[0],s=r[1],u=Object(c.useState)(0),b=Object(f.a)(u,2),p=b[0],x=b[1],O=Object(c.useState)([]),v=Object(f.a)(O,2),g=v[0],m=v[1];function w(){return y.apply(this,arguments)}function y(){return(y=Object(l.a)(o.a.mark((function n(){var a,c,r,s,u,d,l,f;return o.a.wrap((function(n){for(;;)switch(n.prev=n.next){case 0:if(null!=i&&0!==p&&0!==g.length){n.next=2;break}return n.abrupt("return");case 2:return n.next=4,i.predict(j.b([g]));case 4:return r=n.sent,n.next=7,r.data();case 7:if(s=n.sent,u=null===(a=t)||void 0===a?void 0:a.getContext("2d"),d=null===(c=e)||void 0===c?void 0:c.getContext("2d"),null!=u&&null!=d){n.next=12;break}return n.abrupt("return");case 12:for(l=u.createImageData(p,p),f=0;f<p*p;f++)l.data[4*f]=255*s[3*f],l.data[4*f+1]=255*s[3*f+1],l.data[4*f+2]=255*s[3*f+2],l.data[4*f+3]=255;u.putImageData(l,0,0),d.save(),d.scale(d.canvas.width/u.canvas.width,d.canvas.height/u.canvas.height),d.drawImage(t,0,0),d.restore();case 19:case"end":return n.stop()}}),n)})))).apply(this,arguments)}Object(c.useEffect)((function(){Object(l.a)(o.a.mark((function t(){var e;return o.a.wrap((function(t){for(;;)switch(t.prev=t.next){case 0:return t.next=2,j.a("/tfjs/model.json");case 2:e=t.sent,s(e),x(e.outputs[0].shape[1]||0),m(Object(d.a)(Array(e.inputs[0].shape[1]||0)).map((function(t,e){return e<8?0:4*Math.random()-2})));case 6:case"end":return t.stop()}}),t)})))()}),[]),Object(c.useEffect)((function(){w().then()}));return Object(a.jsxs)("div",{children:[Object(a.jsx)("h1",{children:"Idol generator"}),Object(a.jsx)("p",{children:null==i?"Model loading...":"Model loaded."}),Object(a.jsx)("ul",{style:{listStyle:"none",width:"50%",margin:"auto"},children:Object(d.a)(Array(8)).map((function(t,e){return Object(a.jsx)("li",{children:Object(a.jsx)(h.a,{min:-5,max:5,step:.5,defaultValue:0,onChange:function(t){return function(t,e){g[e]=t,w().then()}(t,e)}})},"slider-li-"+e)}))}),Object(a.jsxs)("div",{style:{margin:"8px"},children:[Object(a.jsx)("canvas",{ref:function(e){return t=e},id:"hidden-canvas",height:p+"px",width:p+"px",style:{display:"none"}}),Object(a.jsx)("canvas",{ref:function(t){return e=t},id:"display-canvas",height:"400px",width:"400px",style:{border:"1px solid black"}})]})]})});var p=function(){return Object(a.jsx)("div",{className:"App",children:Object(a.jsx)(b,{})})},x=function(t){t&&t instanceof Function&&n.e(3).then(n.bind(null,307)).then((function(e){var n=e.getCLS,a=e.getFID,c=e.getFCP,r=e.getLCP,i=e.getTTFB;n(t),a(t),c(t),r(t),i(t)}))};s.a.render(Object(a.jsx)(r.a.StrictMode,{children:Object(a.jsx)(p,{})}),document.getElementById("root")),x()}},[[306,1,2]]]);
//# sourceMappingURL=main.26fc7a8f.chunk.js.map