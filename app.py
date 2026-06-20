from flask import Flask, render_template, request, Response, stream_with_context
from chatbot import HELChatBot
import os

class HELWebApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.chatbot = HELChatBot()
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/chat', methods=['POST'])
        def chat():
            data = request.json
            question = data.get('question')
            session_id = data.get('session_id', 'default_session')
            if not question:
                return {"error": "No question provided"}, 400

            def generate():
                for chunk in self.chatbot.ask(question, session_id=session_id):
                    yield chunk

            return Response(stream_with_context(generate()), mimetype='application/json')

        @self.app.route('/like', methods=['POST'])
        def like():
            data = request.json
            sql_id = data.get('sql_id')
            if not sql_id:
                return {"error": "No sql_id provided"}, 400
            self.chatbot.like_answer(sql_id)
            return {"status": "success"}

    def run(self, host='0.0.0.0', port=5000, debug=True):
        self.app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    web_app = HELWebApp()
    web_app.run()
