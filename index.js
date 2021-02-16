var express = require('express');
var app = express();
var http = require('http').Server(app); 
var io = require('socket.io')(http);
var bodyParser = require('body-parser');
var session = require('express-session');
var path = require("path");

var fs = require("fs");
const { strict } = require('assert');

app.set('views', __dirname + '/views');
app.set('view engine', 'ejs');
app.engine('html', require('ejs').renderFile);

var server = http.listen(9407, function(){
 console.log("Express server has started on port 9407")
}); // 생일이 7월이라 임의적으로 설정해봄

app.use(express.static('public'));

app.use(bodyParser.json());
app.use(bodyParser.urlencoded());

var count=1;
io.on('connection', function(socket){ //3
  console.log('connected: ', socket.id);  //3-1

  socket.on('disconnect', function(){ //3-2
    console.log('user disconnected: ', socket.id);
  });

  socket.on('chat question', function(question){ //3-3
    console.log(question);
    //질문이 전송되었다.
    console.log("Socket ID = " + socket.id);
    //질문을 question_socket.id 파일 이름으로 저장한다.
    //먼저 지정한 단어가 있는 질문에 대해서는 그냥 대답한다.
    if(question.includes("언제") &&  question.includes("만들다")) {
      socket.emit('chat answer', "정확히 2021년 2월 15일 오후 10시 14분 테스트가 완료되었습니다.");
    }
    else if(question.includes("이름")){
      socket.emit('chat answer', "제 이름은, 'Bell'입니다.");
    }
    else {
      // answer_socket.id 파일을 읽어서 클라이언트에게 보낸다.
      var filename = './QnA/question_' + socket.id;
      fs.writeFileSync(filename, question, 'utf8');
      // answer_socket.id 파일이 생성 될떄 까지 기다린다.
      var ansfilename = './QnA/answer_' + socket.id;
      var bFound = false;
  
      //1초 간격으로 3초 기다린다.
      var waitTill = new Date(new Date().getTime() + 500);
      while(waitTill > new Date()){}
      if (!fs.existsSync(ansfilename)) {
        var waitTill = new Date(new Date().getTime() + 500);
        while(waitTill > new Date()){}
      }
      if (!fs.existsSync(ansfilename)) {
        var waitTill = new Date(new Date().getTime() + 500);
        while(waitTill > new Date()){}
      }
      if (!fs.existsSync(ansfilename)) {
        var waitTill = new Date(new Date().getTime() + 500);
        while(waitTill > new Date()){}
      }
      if (!fs.existsSync(ansfilename)) {
        var waitTill = new Date(new Date().getTime() + 500);
        while(waitTill > new Date()){}
      }
      if (!fs.existsSync(ansfilename)) {
        var waitTill = new Date(new Date().getTime() + 500);
        while(waitTill > new Date()){}
      }
      if (fs.existsSync(ansfilename)) {
        bFound = true;
      }
      //있으면 파일에서 읽어서 전송하기.
      if (bFound) {
        console.log("answer founded = " + ansfilename);
        var answer = '';
        answer = fs.readFileSync(ansfilename, 'utf8');
        socket.emit('chat answer', answer);  // 해당 질문에 대한 답변을 보낸 유저에게만 전송하도록 한다.
      }
      else {
        console.log("answer Not found = " + ansfilename);
        socket.emit('chat answer', '죄송하지만, 이 데이터는 등록되지 않았습니다.');  // 이거도 똑같음
      }
    }
  });
});
  
var router = require('./router/main')(app, fs);