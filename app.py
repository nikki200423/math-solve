from flask import Flask, render_template, Response
from mathSolver import generate_frames, get_status

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    expr, res = get_status()
    return {"expression": expr, "result": res}

if __name__ == '__main__':
    app.run(debug=True)
