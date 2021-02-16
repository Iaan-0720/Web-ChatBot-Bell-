# Web-ChatBot-Bell-
It can Explain about Words of Standard law from South Korea
- 이것은 대한민국에서 적용되는 법률의 스탠다드 용어들에 대하여 설명하는 챗봇입니다.

# Learning Data
|Title|Contents|From|
|:------:|:---:|:------:|
|Data Name|ChatBot 'Bell'|My Brain|
|Basic Sources From|DeFault ChatBot Data Source|https://github.com/songys/Chatbot_data (송영숙님)|
| 1st Additional Data  |Additional Data Sources for ChatBot system|https://korquad.github.io/ (KorQuAD)|
| 2nd Additional Data  |Legal Terms Data |https://www.data.go.kr/data/15069932/fileData.do (공공데이터 포털)|

# Author
Ian/@Ian (aoa8538@gmail.com)

# Requirement
- 사용된 툴은 다음과 같습니다.
  - Python 3.7.9
  - Tensorflow 2.4.1
  - konlpy
  - pandas
  - numpy
  - socket.io
  
# Structure
- 웹 형태의 챗봇으로써, CSS와 NODE JS도 같이 사용되었으며, 주력 디자인은 'Bootstrap'을 응용했습니다.<br/>
<br/>
├── QnA:&nbsp;&nbsp;&nbsp;&nbsp; 질문과 답변을 저장하는 디렉토리입니다.<br/>
├── socket.io:&nbsp;&nbsp;&nbsp;&nbsp;웹 소켓 구현을 위해 기본적인 파일들이 저장된 디렉토리입니다.<br/>
├── views:&nbsp;&nbsp;&nbsp;&nbsp; 웹 페이지 구현을 위해 작성된 ejs 데이터 파일이 저장된 디렉토리, 'Web Browser'에 해당합니다.<br/>
├── chatbot.py:&nbsp;&nbsp;&nbsp;&nbsp; 프로그램 실행의 시작과 끝까지, 모든 소스가 작성되어 있는 메인 파일입니다.<br/>
├── ChatBotData.csv:&nbsp;&nbsp;&nbsp;&nbsp; '송경숙'님의 데이터 소스(소스 경로는 상위 'Learning Data' 파트에 명시했습니다.)<br/>
├── index.js:&nbsp;&nbsp;&nbsp;&nbsp; 'Web Server'에 해당하는 js 파일입니다.<br/>
├── RealEstate.csv:&nbsp;&nbsp;&nbsp;&nbsp; 부동산(경,공매)및 생활 법률 용어들이 총괄해서 담겨있는 파일입니다.

# Design
![initial](https://user-images.githubusercontent.com/79067558/108025412-c988e100-7069-11eb-8fbd-6903ef0ee0ce.png)<br/>
- 디자인은 상위에 소개된 사진과 같으며, 반드시 'chatbot.py'와 'index.js' 이 두 파일을 모두 실행해주셔야 구동됩니다.<br/>
- 자세한 구동 방식은 'chatbot.py'파일에 수록해놓았으니 참조 바랍니다.<br/>
